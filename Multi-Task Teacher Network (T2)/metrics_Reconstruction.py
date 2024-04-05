import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from skimage.metrics import peak_signal_noise_ratio


def load_masks(folder_path):
    masks = []
    for filename in os.listdir(folder_path):
        mask_path = os.path.join(folder_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    return masks

def calculate_reconstruction_metrics(predicted_mask, ground_truth_mask):
    # Ensure both masks are in the same scale (0 to 1)
    predicted_mask = predicted_mask / 255.0
    ground_truth_mask = ground_truth_mask / 255.0

    # Structural Similarity Index (SSI)
    ssi_index, _ = ssim(predicted_mask, ground_truth_mask, full=True, data_range=1.0)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(predicted_mask, ground_truth_mask)

    # Mean Squared Log Error (MSLE)
    msle = mean_squared_log_error(predicted_mask, ground_truth_mask)

    # Peak Signal-to-Noise Ratio (PSNR)
    psnr = peak_signal_noise_ratio(predicted_mask, ground_truth_mask)

    return ssi_index, mse, msle, psnr


# Paths
predicted_mask_folder = r'D:\Risab\Medical_Experiments\Spleen_TransUNet\all_preds'
ground_truth_mask_folder = r'D:\Risab\Medical_Experiments\Spleen_MultiTask\test_data\image'

# Load masks
predicted_masks = load_masks(predicted_mask_folder)
ground_truth_masks = load_masks(ground_truth_mask_folder)

# Initialize metrics
total_ssi = 0
total_mse = 0
total_msle = 0
total_psnr = 0

# Calculate metrics for each pair of masks
for i in range(len(predicted_masks)):
    predicted_mask = predicted_masks[i]
    ground_truth_mask = ground_truth_masks[i]

    ssi, mse, msle, psnr = calculate_reconstruction_metrics(predicted_mask, ground_truth_mask)

    # Accumulate metrics
    total_ssi += ssi
    total_mse += mse
    total_msle += msle
    total_psnr += psnr

# Calculate average metrics
average_ssi = total_ssi / len(predicted_masks)
average_mse = total_mse / len(predicted_masks)
average_msle = total_msle / len(predicted_masks)
average_psnr = total_psnr / len(predicted_masks)

print("Reconstruction Metrics:")
print(f"SSI (Structural Similarity Index): {average_ssi:.3f}")
print(f"MSE (Mean Squared Error): {average_mse:.3f}")
print(f"MSLE (Mean Squared Log Error): {average_msle:.3f}")
print(f"PSNR (Peak Signal-to-Noise Ratio): {average_psnr:.3f}")
