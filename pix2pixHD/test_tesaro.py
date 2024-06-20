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
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_onnx, run_trt_engine

for i, data in enumerate(dataset):
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
    if opt.export_onnx:
        print('Exporting to ONNX: ', opt.export_onnx)
        assert opt.export_onnx.endswith('onnx'), \
            'Export model file should end with .onnx'
        torch.onnx.export(
            model, [data['label'], data['inst']],
            opt.export_onnx, verbose=True,
        )
        exit(0)
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
                ('input_label',
                 util.tensor2label(data['label'][0], opt.label_nc)),
                ('synthesized_image', util.tensor2im(generated.data[0],imtype=np.uint16)),
            ],
        )

    img_path = data['path']
    print('process image... %s' % img_path)
    img_name = img_path[0].split('/')[-1]
    # Save
    save_path = os.path.join(
            opt.results_dir
    )
#     , opt.name, '%s_%s' %
#             (opt.phase, opt.which_epoch),
#         )

    # Prediction
    
    # for GSK-Broad
#     final_pred_path = os.path.join(save_path,img_name.split('_')[-2],img_name[:-5] + "_virtualstain.tif")
    final_pred_path = os.path.join(save_path,img_name) #[:-7] + "_1.tif")
    
    imsave(final_pred_path, visuals['synthesized_image'])
#     print()
    # Input
#     final_input_path = os.path.join(save_path, img_name[:23]+"_input_label.tiff")
#     imsave(final_input_path, visuals['input_label'])
