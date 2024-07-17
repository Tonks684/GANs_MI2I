import os
from collections import OrderedDict
import torch
import util.my_util as util
from data.data_loader_tesaro import CreateDataLoader
from models.models import create_model
import numpy as np
from options.test_options import TestOptions
from util.my_visualizer import Visualizer
import cv2
from tifffile import imsave
from tqdm import tqdm
from skimage import transform as transform
from util import html

def inference(dataset,opt,model):
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
    for data in tqdm(dataset):
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst'] = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst'] = data['inst'].uint8()
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
            visuals = OrderedDict([('input_label',util.tensor2label(data['label'][0], opt.label_nc)),('synthesized_image', util.tensor2im(generated.data[0],imtype=np.uint16))])
        img_path = data['path']
        print('Processing image... %s' % img_path)
        img_name = img_path[0].split('/')[-1]
        save_path = os.path.join(
                opt.results_dir, img_name)
        imsave(save_path, visuals['synthesized_image'].astype(np.uint16),imagej=True)
    