import os
import shutil

def move_and_rename_files(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through each subfolder in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file is a PNG image
            if file.lower().endswith(".png"):
                # Get the full path of the source file
                source_file_path = os.path.join(root, file)

                # Extract the filename and extension
                _, filename = os.path.split(source_file_path)

                # Remove the word 'predicted' from the filename
                filename = filename.replace("_predicted", "")

                # Create the new file name
                image_name = filename.replace("mask_", "image_")
                destination_file_path = os.path.join(destination_folder, image_name)
                
                # Move and rename the file
                shutil.move(source_file_path, destination_file_path)

if __name__ == "__main__":
    # Set the source and destination folders
    source_folder = r"D:\Risab\Medical_Experiments\Spleen_MultiTask\predicted_output_dekhi"
    destination_folder = r"D:\Risab\Medical_Experiments\Spleen_MultiTask\all_images"

    # Call the function to move and rename files
    move_and_rename_files(source_folder, destination_folder)
