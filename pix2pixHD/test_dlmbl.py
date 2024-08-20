import os
from collections import OrderedDict
import util.util as util
import numpy as np
import cv2
import torch
from tifffile import imsave
from tqdm import tqdm
from skimage import transform as transform
from util import html

def inference(dataset, opt, model):
    """
    Perform inference on the given dataset using the specified model.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to perform inference on.
        opt (argparse.Namespace): The command-line arguments.
        model: The model to use for inference.

    Returns:
        None
    """
    for data in tqdm(dataset):
        generated = model.inference(data['label'], data['inst'], data['image'])
        img_path = data['path']
        # Unnormalize the image
        generated = util.tensors2ims(opt,generated.data,imtype="dlmbl",normalize=False)
        visuals = OrderedDict([('synthesized_image',generated )])
        img_name = img_path[0].split('/')[-1]
        save_path_pred = os.path.join(
                opt.results_dir, img_name)
        imsave(save_path_pred, visuals['synthesized_image'].astype(np.float32),imagej=True)
        
def sampling(dataset, opt, model):
    """
    Perform sampling on the given dataset using the specified model.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to perform sampling on.
        opt (argparse.Namespace): The options for sampling.
        model: The model used for inference.

    Returns:
        None
    """
    os.mkdirs(opt.results_dir, exist_ok=True, parents=True)
    for index, data in tqdm(enumerate(dataset)):
        #Processs only the first 5 data:
        if index == 5:
            break
        stack_pred = np.zeros((opt.variational_inf_runs,512,512))
        for sample in range(opt.variational_inf_runs):
            generated = model.inference(data['label'], data['inst'], data['image'])
            visuals = OrderedDict([('input_label',util.tensors2ims(opt, data['label'][0], imtype=np.float32,normalize=False)),('synthesized_image', util.tensors2ims(opt,generated.data[0],imtype='dlmbl',normalize=False))])
            stack_pred[sample] = visuals['synthesized_image']

        img_path = data['path']
        img_name = img_path[0].split('/')[-1][:-5]
        samples_name = f"{img_name}_samples.tiff"
        save_path = os.path.join(
                opt.results_dir, samples_name)
        imsave(save_path, stack_pred.astype(np.float32),imagej=True)
