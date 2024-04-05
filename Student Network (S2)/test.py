import os
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    try:
        y_true = y_true.cpu().numpy().flatten() > 0.5
        y_pred = y_pred.cpu().numpy().flatten() > 0.5

        score_jaccard = jaccard_score(y_true, y_pred, zero_division=1)
        score_f1 = f1_score(y_true, y_pred, zero_division=1)
        score_recall = recall_score(y_true, y_pred, zero_division=1)
        score_precision = precision_score(y_true, y_pred, zero_division=1)
        score_acc = accuracy_score(y_true, y_pred)

        return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

    except Warning as e:
        print(f"Warning: {e}")
        return [0.0, 0.0, 0.0, 0.0, 0.0]

def mask_parse(mask):
    return np.expand_dims(mask, axis=-1) // 255  # Ensure it's a single-channel binary mask

if __name__ == "__main__":
    seeding(42)
    create_dir("results")

    test_x = sorted(glob(r'C:\Users\user\Desktop\Spleen_Seg\test_data\image\*'))
    test_y = sorted(glob(r'C:\Users\user\Desktop\Spleen_Seg\test_data\mask\*'))

    H, W = 256, 256
    size = (W, H)
    checkpoint_path = r'C:\Users\user\Desktop\Spleen_Seg\files\checkpoint.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = np.zeros(5)
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.splitext(os.path.basename(x))[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = np.transpose(image, (2, 0, 1)) / 255.0
        x = np.expand_dims(x, axis=0).astype(np.float32)
        x = torch.from_numpy(x).to(device)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = np.expand_dims(mask, axis=0) / 255.0
        y = np.expand_dims(y, axis=0).astype(np.float32)
        y = torch.from_numpy(y).to(device)

        with torch.no_grad():
            start_time = time.time()
            pred_y = torch.sigmoid(model(x))
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score += np.array(score)

            pred_y = (pred_y > 0.5).cpu().numpy().squeeze().astype(np.uint8)

        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)

        # Ensure both masks have the same number of channels
        ori_mask = ori_mask[:, :, 0:1]  # Take only the first channel

        line = np.ones((size[1], 10, 3)) * 128
        cat_images = np.concatenate([image, line, ori_mask, line, pred_y * 255], axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)

    metrics_score /= len(test_x)
    jaccard, f1, recall, precision, acc = metrics_score
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)

