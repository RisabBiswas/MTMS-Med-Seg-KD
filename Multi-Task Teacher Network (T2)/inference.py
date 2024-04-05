import os
from glob import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

H, W = 256, 256
size = (W, H)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define a function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inference(model, image_path, output_folder):
    # Load the image and preprocess it
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Move the model and input data to the appropriate device
    model = model.to(device)
    img_tensor = img_tensor.to(device)

    # Perform inference for segmentation and reconstruction
    model.eval()
    with torch.no_grad():
        segmentation_output, reconstruction_output = model(img_tensor)

    # Convert the segmentation output to a binary mask
    predicted_mask = (segmentation_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Save the predicted segmentation mask as PNG directly in the "predicted_output" folder
    os.makedirs(os.path.join(output_folder, "segmentation"), exist_ok=True)
    mask_filename = os.path.join(output_folder, "segmentation", os.path.basename(image_path).replace('image', 'mask').replace('.png', '_predicted.png'))
    Image.fromarray((predicted_mask * 255).astype(np.uint8)).save(mask_filename)

    # Save the predicted reconstruction as an image
    reconstruction_image = (reconstruction_output.squeeze().permute(1, 2, 0).cpu().numpy())
    reconstruction_image = (reconstruction_image * 255).astype(np.uint8)

    os.makedirs(os.path.join(output_folder, "reconstruction"), exist_ok=True)
    reconstruction_filename = os.path.join(output_folder, "reconstruction", os.path.basename(image_path).replace('image', 'reconstruction').replace('.png', '_predicted.png'))

    Image.fromarray(reconstruction_image).save(reconstruction_filename)

# Paths to test data
test_x = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_MultiTask\test_data\image\*'))
test_y = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_MultiTask\test_data\mask\*'))

# Load your pre-trained model
from model_transunet import TransUnet  # Adjust based on your actual model import
model = TransUnet(in_channels=3, img_dim=256, vit_blocks=8, vit_dim_linear_mhsa_block=2048, classes=3)  # Replace with the actual model instantiation code

checkpoint_dir = r'D:\Risab\Medical_Experiments\Spleen_TransUNet\Model_Weights_MultiTask'
checkpoint_path = os.path.join(checkpoint_dir, 'MultiTask_checkpoint_v1.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Count the parameters
total_params = count_parameters(model)
print("Total number of parameters: ", total_params)

# Output folder for predicted masks and reconstructions
output_folder = "predicted_output_dekhi"

# Inference loop
for image_path, mask_path in zip(test_x, test_y):
    # Create a unique folder for each test image
    image_folder_name = os.path.basename(image_path).replace('.png', '')
    current_output_folder = os.path.join(output_folder, image_folder_name)

    # Perform inference for segmentation and reconstruction
    inference(model, image_path, current_output_folder)

print("Inference completed.")


