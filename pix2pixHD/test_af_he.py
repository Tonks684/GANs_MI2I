import os
from collections import OrderedDict
import torch
import util.my_util as util
from data.data_loader_af_he import CreateDataLoader
from models.models import create_model
import numpy as np
from options.test_options import TestOptions
from util.my_visualizer import Visualizer
import cv2
from tifffile import imsave

from skimage import transform as transform
from util import html


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(
#     opt.results_dir, opt.name, '%s_%s' %
#     (opt.phase, opt.which_epoch),
# )
# webpage = html.HTML(
#     web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
#     (opt.name, opt.phase, opt.which_epoch),
# )

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    # if opt.data_type == 16:
    #     model.half()
    # elif opt.data_type == 8:
    #     model.type(torch.uint8)
    if opt.verbose:
        print(model)
else:
    from run_engine import run_onnx, run_trt_engine

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    # if opt.data_type == 16:
    #     data['label'] = data['label'].half()
    #     data['inst'] = data['inst'].half()
    # if opt.data_type == 8:
    #     data['af_cy3'] = data['af_cy3'].to(torch.uint8)
    #     data['af_dapi'] = data['af_dapi'].to(torch.uint8)
    #     data['image'] = data['image'].to(torch.uint8)
    #     # data['label'] = data['label'].to(torch.uint8)
        # data['inst']  = data['inst'].to(torch.uint8)
    # if opt.export_onnx:
    #     print('Exporting to ONNX: ', opt.export_onnx)
    #     assert opt.export_onnx.endswith('onnx'), \
    #         'Export model file should end with .onnx'
    #     torch.onnx.export(
    #         model, [data['label'], data['inst']],
    #         opt.export_onnx, verbose=True,
    #     )
    #     exit(0)
    minibatch = 1
    if opt.engine:
        generated = run_trt_engine(
            opt.engine, minibatch, [data['label'], data['inst']],
        )
    elif opt.onnx:
        generated = run_onnx(
            opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']],
        )
    else:
        generated = model.inference(data['af_cy3'], data['af_dapi'], data['image'])

    generated.cpu()
    if opt.output_reshape:
        prediction = util.tensor2im(generated.data[0])
        prediction = cv2.resize(
            prediction, (opt.output_reshape, opt.output_reshape),
            interpolation=cv2.INTER_LINEAR)
        input_label = util.tensor2label(data['label'][0], opt.label_nc)
        input_label = transform.resize(
            input_label, (opt.output_reshape, opt.output_reshape))
        visuals = OrderedDict(
             [
                 ('input_label', input_label),
                 ('synthesized_image', prediction),
             ],
         )
        print("Min orig {} Max orig {}".format(
            np.min(util.tensor2im(generated.data[0])),
            np.max(util.tensor2im(generated.data[0]))))
        print("Min reshape {} Max reshape {}".format(np.min(prediction),
                                                     np.max(prediction)))

    else:
        visuals = OrderedDict(
            [
                ('af_cy3',
                 util.tensor2label(data['af_cy3'][0], opt.label_nc,imtype=np.float32)),
                ('af_dapi',
                 util.tensor2label(data['af_dapi'][0], opt.label_nc,imtype=np.float32)),
                ('synthesized_image', 
                util.tensor2im(generated.data[0],imtype=np.uint8)),
            ],
        )
    print(visuals['synthesized_image'].max())
    
    img_path = data['path']
    # print('process image... %s' % img_path)
    img_name = img_path[0].split('/')[-1]
    # Save
    save_path = os.path.join(
            opt.results_dir, opt.name, '%s_%s' %
            (opt.phase, opt.which_epoch),
        )

    # Prediction
    # final_pred_path = os.path.join(save_path, img_name[:-5] + "_synthesized_image.tiff")
    # imsave(final_pred_path, visuals['synthesized_image'].astype(np.float32))
    final_pred_path = os.path.join(save_path, img_name[:-5] + "_synthesized_image8bit.tiff")
    imsave(final_pred_path, visuals['synthesized_image'].astype(np.uint8),imagej=True)
    # Inputs
    
    # final_input_path = os.path.join(save_path, img_name[:-5]+"_af_dapi.tiff")
    # imsave(final_input_path, visuals['af_dapi'].astype(np.float32))
    # final_input_path = os.path.join(save_path, img_name[:-5]+"_af_cy3.tiff")
    # imsave(final_input_path, visuals['af_cy3'].astype(np.float32))
