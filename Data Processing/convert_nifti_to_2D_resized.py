import os
import numpy as np
from PIL import Image
import nibabel as nib

def convert_nifti_to_2D():
    data_dir = r'D:\Risab\Medical_Experiments\Spleen_Segmentation\Dataset\Task09_Spleen\imagesTr'
    data_dir_mask = r'D:\Risab\Medical_Experiments\Spleen_Segmentation\Dataset\Task09_Spleen\labelsTr'

    subjects = [f'spleen_{x}' for x in range(2, 90)]
    print(f'There are {len(subjects)} subjects in total')

    output_dir = r'D:\Risab\Medical_Experiments\Spleen_Segmentation\Dataset\Task09_Spleen\output2d_images'
    output_dir_mask = r'D:\Risab\Medical_Experiments\Spleen_Segmentation\Dataset\Task09_Spleen\output2d_masks'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)

    for subject in subjects:
        print(f'Processing {subject}')

        sa_zip_file = os.path.join(data_dir, f'{subject}.nii.gz')
        sa_zip_file_mask = os.path.join(data_dir_mask, f'{subject}.nii.gz')

        if not os.path.exists(sa_zip_file) or not os.path.exists(sa_zip_file_mask):
            print(f"Skipping {subject} - NIfTI file or mask not found.")
            continue

        img = nib.load(sa_zip_file)
        data = np.array(img.get_fdata())

        img_mask = nib.load(sa_zip_file_mask)
        data_mask = np.array(img_mask.get_fdata())

        multiplier = 255.0 / data.max() if data.max() > 0 else 1.0
        print(f'max_pixel_value = {data.max()}, multiplier = {multiplier}')

        for s in range(data.shape[2]):
            image_filename = os.path.join(output_dir, f'image_{subject}_{str(s).zfill(2)}.png')
            mask_filename = os.path.join(output_dir_mask, f'mask_{subject}_{str(s).zfill(2)}.png')
            Image.fromarray(data[:, :, s]).convert("RGB").save(image_filename)
            Image.fromarray(data_mask[:, :, s]*255).convert("RGB").save(mask_filename)

if __name__ == '__main__':
    convert_nifti_to_2D()