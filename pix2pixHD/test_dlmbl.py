import os
from collections import OrderedDict
import torch
import util.util as util
from data.data_loader_dlmbl import CreateDataLoader
from models.models import create_model
import numpy as np
from options.test_options import TestOptions
from util.visualizer import Visualizer
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
        visuals = OrderedDict([('synthesized_image', util.tensor2im(opt,generated.data[0],imtype="dlmbl",normalize=False))])
        img_name = img_path[0].split('/')[-1]
        save_path_pred = os.path.join(
                opt.results_dir, img_name)
        save_path_real = os.path.join(
                opt.results_dir, img_path[0].split('/')[-1][:-5] + '_real.tiff')
        print('Processing image... %s' % img_path)
        print(generated.data[0].shape, data['image'].shape)
        imsave(save_path_pred, visuals['synthesized_image'].astype(np.float32),imagej=True)
        imsave(save_path_real,util.tensor2im(opt,data['image'],imtype='dlmbl',normalize=False),imagej=True)
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
    for data in tqdm(dataset):
        for sample in range(opt.variational_inf_runs):
            for data in tqdm(dataset):
                generated = model.inference(data['label'], data['inst'], data['image'])
                generated.cpu()
                if opt.output_reshape:
                    prediction = util.tensor2im(generated.data[0], imtype=np.uint16)
                    prediction = cv2.resize(
                        prediction, (opt.output_reshape, opt.output_reshape),
                        interpolation=cv2.INTER_LINEAR)
                    input_label = util.tensor2label(data['label'][0], opt.label_nc)
                    input_label = transform.resize(
                        input_label, (opt.output_reshape, opt.output_reshape))
                    visuals = OrderedDict([('input_label', input_label),('synthesized_image', prediction)])
                else:
                    visuals = OrderedDict([('input_label',util.tensor2label(data['label'][0], opt.label_nc)),('synthesized_image', util.tensor2im(generated.data[0],imtype='dlmbl',normalize=False))])
                img_path = data['path']
                print('Processing image... %s' % img_path)
                img_name = img_path[0].split('/')[-1]
                image_name = f"img_name_{sample}.tiff"
                save_path = os.path.join(
                        opt.results_dir, img_name)
                imsave(save_path, visuals['synthesized_image'].astype(np.float32),imagej=True)
