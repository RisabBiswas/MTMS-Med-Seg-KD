import os
import cv2
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score

def load_masks(folder_path):
    masks = []
    for filename in os.listdir(folder_path):
        mask_path = os.path.join(folder_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    return masks

def binary_conversion(mask):
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return binary_mask

def calculate_metrics(predicted_mask, ground_truth_mask):
    predicted_flat = predicted_mask.flatten()
    ground_truth_flat = ground_truth_mask.flatten()

    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum(np.logical_and(predicted_flat == 255, ground_truth_flat == 255))
    fp = np.sum(np.logical_and(predicted_flat == 255, ground_truth_flat == 0))
    fn = np.sum(np.logical_and(predicted_flat == 0, ground_truth_flat == 255))

    return tp, fp, fn

# Paths
predicted_mask_folder = r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student_(SmallerUNet)\predicted_mask_KD_B-B+D-D+PMD'
ground_truth_mask_folder = r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student_(SmallerUNet)\test_data\mask'

# Load masks
predicted_masks = load_masks(predicted_mask_folder)
ground_truth_masks = load_masks(ground_truth_mask_folder)

# Initialize counters
total_tp = 0
total_fp = 0
total_fn = 0

# Calculate metrics for each pair of masks
for i in range(len(predicted_masks)):
    predicted_mask_binary = binary_conversion(predicted_masks[i])
    ground_truth_mask_binary = binary_conversion(ground_truth_masks[i])

    tp, fp, fn = calculate_metrics(predicted_mask_binary, ground_truth_mask_binary)

    # Accumulate counts
    total_tp += tp
    total_fp += fp
    total_fn += fn

# Calculate overall metrics
precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
jaccard = total_tp / (total_tp + total_fp + total_fn)
f1 = 2 * (precision * recall) / (precision + recall)
iou = total_tp / (total_tp + total_fp + total_fn)
dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn)  # Dice coefficient

print("Overall Metrics:")
print(f"IoU (Intersection over Union): {iou}")
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Dice Coefficient: {dice}")