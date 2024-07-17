import numpy as np
import zarr
import os
from tqdm import tqdm
import random
from tifffile import imread,imsave
import os
import random
import shutil
from pathlib import Path
import argparse
import shutil

def crop_image(image, crop_size=512):
    """
    Crop the given image into multiple crops of the specified size.

    Args:
        image (numpy.ndarray): The input image to be cropped.
        crop_size (int, optional): The size of each crop. Defaults to 512.

    Returns:
        list: A list of cropped images.

    """
    crops = []
    height, width = image.shape[:2]
    
    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            crop = image[i:i+crop_size, j:j+crop_size]
            crops.append(crop)
    
    return crops

def copy_files(file_list, split_type, folders):
    """
    Copy files from the given file list to the specified split type and folders.

    Args:
        file_list (list): A list of file names to be copied.
        split_type (str): The split type to which the files should be copied.
        folders (dict): A dictionary containing folder types as keys and folder paths as values.

    Returns:
        None
    """
    for file_name in file_list:
        for folder_type, folder_path in folders.items():
            src = folder_path / file_name
            dst = folder_path / split_type / file_name
            shutil.copy(src, dst)

def create_train_val_split(input_folder, nuclei_folder, cyto_folder, split_ratio=0.8):
    """
    Create train and validation splits of the dataset.

    Args:
        input_folder (str): Path to the input folder containing the images.
        nuclei_folder (str): Path to the nuclei folder where the train and validation splits will be created.
        cyto_folder (str): Path to the cyto folder where the train and validation splits will be created.
        split_ratio (float, optional): Ratio of the dataset to be used for training. Defaults to 0.8.
    """

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
    
    # Copy train and val files
    copy_files(train_files, 'train',folders)
    copy_files(val_files, 'val',folders)

def save_crops(input_crops, channel_folder, a549_hoechst_folder, img, args):
    """
    Save the input crops as TIFF images.

    Args:
        input_crops (list): List of input crops to be saved.
        channel_folder (str): Name of the channel folder.
        a549_hoechst_folder (str): Name of the A549 Hoechst folder.
        img (str): Name of the image.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    for i, crop in enumerate(input_crops, args.crop_size):
        output_path = os.path.join(args.output_image_folder, channel_folder, f"{a549_hoechst_folder}_{img}_crop{i}.tiff")
        imsave(output_path, crop.astype(np.float32), imagej=True)
        print(f"Saved crop to {output_path}")
    

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--output_image_folder", type=str, required=True)
    args.add_argument("--crop_size", type=int, default=512)
    args.parse_args()
    #Extract train and validation
    for a549_hoechst_folder in tqdm(range(1,30)):
        train_dataset_url = \
        f"https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/training/a549_hoechst_cellmask_train_val.zarr/0/0/{a549_hoechst_folder}/0"
        try:
            dataset = zarr.open(train_dataset_url, mode='r')
            print("Remote dataset accessed successfully.")
            dataset_np = np.array(dataset)
            for img in tqdm(range(dataset_np.shape[2])):
                input_x = dataset_np[0,0,img]
                input_crops = crop_image(input_x)
                save_crops(input_crops, 'input', a549_hoechst_folder, img, args)
                nuclei = dataset_np[0,1,img]
                nuclei_crops = crop_image(nuclei)
                save_crops(nuclei_crops, 'nuclei',a549_hoechst_folder, img, args)
                cyto = dataset_np[0,2,img]
                cyto_crops = crop_image(cyto,args.crop_size)
                save_crops(cyto_crops,'cyto', a549_hoechst_folder, img, args)
        except Exception as e:
            print(f"Failed to access remote dataset: {e}")

    create_train_val_split(f'{args.output_image_folder}/input',f'{args.output_image_folder}/nuclei',f'{args.output_image_folder}/cyto')
    # Extract test set
    for a549_hoechst_folder in tqdm([0,5,10,15,20,25,30]):
        test_dataset_url = \
        f"https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/0/0/{a549_hoechst_folder}/0/"
        try:
            dataset = zarr.open(train_dataset_url, mode='r')
            print("Remote dataset accessed successfully.")
            dataset_np = np.array(dataset)
            for img in tqdm(range(dataset_np.shape[2])):
                input_x = dataset_np[0,0,img]
                input_crops = crop_image(input_x)
                save_crops(input_crops,'input/test/', a549_hoechst_folder, img, args)
                nuclei = dataset_np[0,1,img]
                nuclei_crops = crop_image(nuclei)
                save_crops(nuclei_crops,'nuclei/test/', a549_hoechst_folder, img, args)
                cyto = dataset_np[0,2,img]
                cyto_crops = crop_image(cyto,'cyto/test/',args.crop_size)
                save_crops(cyto_crops, a549_hoechst_folder, img, args)
        except Exception as e:
            print(f"Failed to access remote dataset: {e}")