import os
from glob import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

H, W = 256, 256
size = (W, H)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch

# Define a function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Function to perform inference
def inference(model, image_path, output_folder):
    # Load the image and preprocess it
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Move the model and input data to the appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img = img.to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
         _, _, _, output = model(img)

    # Convert the output to a binary mask
    predicted_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # with torch.no_grad():
    #     _, output = model(img)
    #     print(output.shape)
    # # Assuming the model returns a tuple, extract the relevant tensor
    # output_tensor = output[0] if isinstance(output, tuple) else output

    # # Convert the output to a binary mask
    # predicted_mask = (output_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)



    # Save the predicted mask as PNG directly in the "predicted_mask" folder
    os.makedirs(output_folder, exist_ok=True)
    mask_filename = os.path.join(output_folder, os.path.basename(image_path).replace('image', 'mask').replace('.png', '_predicted.png'))
    Image.fromarray((predicted_mask * 255).astype(np.uint8)).save(mask_filename)

# Paths to test data
test_x = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_KD_Student\test_data\image\*'))
test_y = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_KD_Student\test_data\mask\*'))

# Load your pre-trained model
from model import build_small_unet  # Adjust based on your actual model import
model = build_small_unet().to(device)  # Replace with the actual model instantiation code
checkpoint_path = r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student\Modelfiles_KD_E-E+PMD\KD_E-E+PMD_Checkpoint.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Count the parameters
total_params = count_parameters(model)
print("Total number of parameters: ", total_params)

# Output folder for predicted masks
output_folder = "predicted_mask_KD_E-E+PMD"

# Inference loop
for image_path, mask_path in zip(test_x, test_y):
    # Perform inference and save the predicted mask directly in the output folder
    inference(model, image_path, output_folder)

print("Inference Completed.")
