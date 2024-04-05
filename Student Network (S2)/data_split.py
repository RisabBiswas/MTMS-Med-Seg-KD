import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_organize_data(input_dir, output_train_dir, output_test_dir, test_size=0.2, random_state=42):
    # List the subdirectories within the input directory
    subdirectories = [subdir for subdir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, subdir))]

    # Create train and test directories if they don't exist
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    # Create 'images' and 'masks' folders within train and test directories
    for data_dir in [output_train_dir, output_test_dir]:
        os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'masks'), exist_ok=True)

    # Perform the split for each subdirectory (e.g., 'images', 'masks')
    for subdir in subdirectories:
        print(f"Processing {subdir}...")

        # Get a list of all files in the current subdirectory
        all_files = os.listdir(os.path.join(input_dir, subdir))

        # Split the files into train and test sets
        train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=random_state)

        # Copy train files to the train directory
        for file in train_files:
            src_path = os.path.join(input_dir, subdir, file)
            dest_path = os.path.join(output_train_dir, subdir, file)
            shutil.copy(src_path, dest_path)

        # Copy test files to the test directory
        for file in test_files:
            src_path = os.path.join(input_dir, subdir, file)
            dest_path = os.path.join(output_test_dir, subdir, file)
            shutil.copy(src_path, dest_path)

if __name__ == '__main__':
    input_data_dir = r'C:\Users\user\Desktop\Spleen_Seg\data'
    output_train_data_dir = r'C:\Users\user\Desktop\Spleen_Seg\train_data'
    output_test_data_dir = r'C:\Users\user\Desktop\Spleen_Seg\test_data'

    # Perform the 80-20 split and organize the data
    split_and_organize_data(input_data_dir, output_train_data_dir, output_test_data_dir, test_size=0.2, random_state=42)
