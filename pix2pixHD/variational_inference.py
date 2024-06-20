
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

from skimage import transform as transform
from util import html



def crop(img,crop=256,type=None):

    """
    img: image to be cropped
    crop: crop size
    type: center or random
    """
    #Dimension of input array
    width, height = img.shape
    # Generate random cordinates for the crop
    max_y = height - crop
    max_x = max_y
    if type == 'random':
        start_y = np.random.randint(0,max_y +1)
        start_x = np.random.randint(0,max_x +1)
        end_y = start_y + crop
        end_x = start_x + crop
    elif type == 'center':
        start_x = (width - crop)//2
        start_y = (height - crop)//2
        end_y = height - start_y
        end_x = width - start_x
    else:
        raise ValueError(f'unknown crop type {type}')

    # Crop array using slicing
    crop_array = img[start_x:end_x, start_y:end_y]
    return crop_array




opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

if not os.path.exists(opt.variational_inf_path):
    os.makedirs(opt.variational_inf_path)
    print('Successfully create the directory %s' % opt.variational_inf_path)

# Inference
# Model creation
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

# for each image
for i, data in enumerate(dataset):
    if not os.path.exists(f'{opt.variational_inf_path}/{i}/'):
        os.makedirs(f'{opt.variational_inf_path}/{i}/')

    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst'] = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst'] = data['inst'].uint8()
        # data['label'] = data['label'].to(torch.uint8)
        # data['inst']  = data['inst'].to(torch.uint8)
    for run in range(opt.variational_inf_runs):
        minibatch = 1
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch,
                                       [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch,
                                 [data['label'], data['inst']])
        else:
            generated = model.inference(data['label'],
                                        data['inst'], data['image'])
        generated.cpu()
        # Reshape if needed
        prediction = util.tensor2im(generated.data[0],imtype=np.uint16)
        if opt.output_reshape:
            print("Initial Prediction Shape {}".format(prediction.shape))
            prediction = cv2.resize(
                prediction, (opt.output_reshape, opt.output_reshape),
                interpolation=cv2.INTER_LINEAR)
        crop_prediction = crop(prediction,256,'center')   
        img_path = data['path']
        img_name = img_path[0].split('/')[-1][:-8]
        img_name_final = f'{img_name}_synthesized_{run}.tiff'
        output_path = f'{opt.variational_inf_path}/{img_name}/{img_name_final}'
        if not os.path.exists(f'{opt.variational_inf_path}/{img_name}/'):
            os.mkdir(f'{opt.variational_inf_path}/{img_name}/')
       
        imsave(output_path, crop_prediction.astype(np.uint16))
        print('Successfully Saved Sample at {} '.format(output_path))
        