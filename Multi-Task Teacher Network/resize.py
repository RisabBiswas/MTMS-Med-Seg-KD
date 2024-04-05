import os
import cv2
from glob import glob

def resize_images_and_masks(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, target_size):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    image_paths = glob(os.path.join(input_image_folder, '*'))

    for image_path in image_paths:
        # Resize images
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # Extract the file name and extension
        _, filename = os.path.split(image_path)
        output_image_path = os.path.join(output_image_folder, filename)

        cv2.imwrite(output_image_path, resized_img)

        # Resize masks
        mask_path = os.path.join(input_mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)

        output_mask_path = os.path.join(output_mask_folder, filename)
        cv2.imwrite(output_mask_path, resized_mask)

if __name__ == "__main__":
    train_x_input_folder = r'C:\Users\user\Desktop\Spleen_Seg\train_data\image'
    train_y_input_folder = r'C:\Users\user\Desktop\Spleen_Seg\train_data\mask'
    train_x_output_folder = r'C:\Users\user\Desktop\Spleen_Seg\train_data\resized_images'
    train_y_output_folder = r'C:\Users\user\Desktop\Spleen_Seg\train_data\resized_masks'
    target_size = (256, 256)  # Change this to your desired size

    resize_images_and_masks(train_x_input_folder, train_y_input_folder, train_x_output_folder, train_y_output_folder, target_size)

    valid_x_input_folder = r'C:\Users\user\Desktop\Spleen_Seg\test_data\image'
    valid_y_input_folder = r'C:\Users\user\Desktop\Spleen_Seg\test_data\mask'
    valid_x_output_folder = r'C:\Users\user\Desktop\Spleen_Seg\test_data\resized_images'
    valid_y_output_folder = r'C:\Users\user\Desktop\Spleen_Seg\test_data\resized_masks'

    resize_images_and_masks(valid_x_input_folder, valid_y_input_folder, valid_x_output_folder, valid_y_output_folder, target_size)
