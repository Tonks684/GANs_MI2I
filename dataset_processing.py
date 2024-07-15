import numpy as np
import zarr
import os
from tqdm import tqdm
from tifffile import  imsave

def crop_image(image):
    crops = []
    height, width = image.shape[:2]
    crop_size = 512
    
    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            crop = image[i:i+crop_size, j:j+crop_size]
            crops.append(crop)
    
    return crops

if __name__ == "__main__":
    
    output_image_folder = "/hpc/projects/upt/samuel_tonks_experimental_space/datasets/a549_hoechst/"
    for a549_hoechst_folder in tqdm(range(1,29)):
        train_dataset_url = \
        f"https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/training/a549_hoechst_cellmask_train_val.zarr/0/0/{a549_hoechst_folder}/0"
        try:
            dataset = zarr.open(train_dataset_url, mode='r')
            print("Remote dataset accessed successfully.")
            dataset_np = np.array(dataset)
            for img in tqdm(range(dataset_np.shape[2])):
                input = dataset_np[0,0,img]
                input_crops = crop_image(input)
                for i, crop in enumerate(input_crops):
                    output_path = os.path.join(output_image_folder,'input', f"{a549_hoechst_folder}_{img}_crop{i}.tiff")
                    imsave(output_path, crop.astype(np.float32),imagej=True)
                    print(f"Saved crop to {output_path}")
                nuclei = dataset_np[0,1,img]
                nuclei_crops = crop_image(nuclei)
                for i, crop in enumerate(nuclei_crops):
                    output_path = os.path.join(output_image_folder,'nuclei', f"{a549_hoechst_folder}_{img}_crop{i}.tiff")
                    imsave(output_path, crop.astype(np.float32),imagej=True)
                    print(f"Saved crop to {output_path}")
                cyto = dataset_np[0,2,img]
                cyto_crops = crop_image(cyto)
                for i, crop in enumerate(cyto_crops):
                    output_path = os.path.join(output_image_folder,'cyto', f"{a549_hoechst_folder}_{img}_crop{i}.tiff")
                    imsave(output_path, crop.astype(np.float32),imagej=True)
                    print(f"Saved crop to {output_path}")
                


        except Exception as e:
            print(f"Failed to access remote dataset: {e}")
        
