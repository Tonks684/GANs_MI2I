import os
import random
from tifffile import imread,imsave
import os
import random
import shutil
from pathlib import Path

def create_train_val_split(input_folder, nuclei_folder, cyto_folder, split_ratio=0.8):
    # Define paths
    folders = {
        'input': Path(input_folder),
        'nuclei': Path(nuclei_folder),
        'cyto': Path(cyto_folder)
    }
    
    # Create train and val folders
    for folder_type in folders.values():
        (folder_type / 'train').mkdir(parents=True, exist_ok=True)
        (folder_type / 'val').mkdir(parents=True, exist_ok=True)
    
    # Get list of all images (assuming all folders have the same images)
    image_files = [f for f in os.listdir(folders['input']) if os.path.isfile(folders['input'] / f)]
    
    # Shuffle images
    random.shuffle(image_files)
    
    # Calculate split index
    split_index = int(len(image_files) * split_ratio)
    
    # Split into train and val
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    # Function to copy files to respective folders
    def copy_files(file_list, split_type):
        for file_name in file_list:
            for folder_type, folder_path in folders.items():
                src = folder_path / file_name
                dst = folder_path / split_type / file_name
                shutil.copy(src, dst)
    
    # Copy train and val files
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

if __name__ == "__main__":
    root_dir = f"/hpc/projects/upt/samuel_tonks_experimental_space/datasets/a549_hoechst/"
    create_train_val_split(f'{root_dir}/input',f'{root_dir}/nuclei',f'{root_dir}/cyto')
    print("Done")