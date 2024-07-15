import os
import random
from tifffile import imread,imsave
# Set the path to the directory containing the images
total_indexes = [i for i in range(3311)]
train_index = [random.randint(0,3311) for i in range(2650)]
test_index = [i for i in total_indexes if i not in train_index]
modes = ['nuclei', 'input', 'cyto']
for mode in modes:
    # Set the paths for the training and testing directories
    image_dir = f"/hpc/projects/upt/samuel_tonks_experimental_space/datasets/a549_hoechst/{mode}"
    train_dir = f"{image_dir}/train"
    test_dir = f"{image_dir}/val"

    # Create the training and testing directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tiff")]

    # Split the image files into training and testing sets
    train_files = [image_files[i] for i in train_index]
    test_files = [image_files[i] for i in test_index]
    
    for train, test in zip(train_files, test_files):
        train_tmp = imread(train)
        train_name = train.split('/')[-1]
        test_tmp = imread(test)
        test_name = test.split('/')[-1]
        imsave(f"{train_dir}/train/{train}", train_tmp.astype(np.uint16),imagej=True)
        imsave(f"{test_dir}/train/{test}", test_tmp.astype(np.uint16),imagej=True)