import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json
from options.train_options import TrainOptions
from models.models import create_model
from data.my_data_loader import CreateDataLoader
import util.my_util as util
from util.my_visualizer import Visualizer

def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
opt.no_vgg_loss = True
# opt.ndf = '32'
print(f'VGG loss {opt.no_vgg_loss}\nNDF {opt.ndf}')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

# Load Train Set for input into model
data_loader = CreateDataLoader(opt)
dataset_train = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# Load Val Set
opt.phase = 'val'
print(opt.phase)
data_loader = CreateDataLoader(opt)
dataset_val = data_loader.load_data()
val_size = len(data_loader)
print('#validation images = %d' % val_size)
opt.phase = 'train'

model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
print("Model Defined")
total_steps = (start_epoch-1) * dataset_size + epoch_iter
print("Total Steps {}".format(total_steps))
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    ssim_scores, psnr_scores = [], []
    for i, data in enumerate(dataset_train, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']),
            Variable(data['image']), Variable(data['feat']), infer=True) #  infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()
        else:
            loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()
        else:
            loss_D.backward()
        optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ### display output images
        imtype = np.uint8 if opt.data_type == '8' else np.uint16
        visuals = OrderedDict([('input_label',
                                util.tensor2label(data['label'][0],
                                                  opt.label_nc)),
                               ('synthesized_image',
                                util.tensor2im(generated.data[0],imtype)),
                               ('real_image', 
                               util.tensor2im(data['image'][0],imtype))])
        
        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        # Compute metric
        
        gen_image = visuals['synthesized_image']
        gt_image = visuals['real_image']
        print('------------------------------------------------')
        print(generated.data[0].shape)
        # print(gen_image.shape)
        if gt_image.shape[2] == 3: # RGB
            tmp_ssim = [] 
            tmp_psnr = []
            for i in range(0,3):
                channel_score_ssim = ssim(gt_image[:,:,i], gen_image[:,:,i])
                tmp_ssim.append(channel_score_ssim)
                channel_score_psnr = psnr(gt_image[:,:,i], gen_image[:,:,i])
                tmp_psnr.append(channel_score_psnr)
                # print(channel_score_psnr)
            score_ssim = np.mean(tmp_ssim)
            score_psnr = np.mean(tmp_psnr)
            
        else:
            score_ssim = ssim(gt_image, gen_image)

        ssim_scores.append(score_ssim)
        psnr_scores.append(score_psnr)

        if epoch_iter >= dataset_size:
                break
    avg_ssim_train = np.mean(ssim_scores)
    print('Averagae SSIM for Epoch {} Train = {}'.format(epoch,
                                                             avg_ssim_train))
    avg_psnr_train = np.mean(psnr_scores)
    print('Averagae PSNR for Epoch {} Train = {}'.format(epoch,
                                                           avg_psnr_train))
    ### Compare with validation set and save metrics
    ############## Forward Pass Val ######################
    ssim_scores, psnr_scores = [], []
    for data in dataset_val:
        _, generated = model(Variable(data['label']),
                             Variable(data['inst']),
                             Variable(data['image']),
                             Variable(data['feat']), infer=True)
        visuals_val = OrderedDict([('input_label',
                                util.tensor2label(data['label'][0],
                                                  opt.label_nc)),
                               ('synthesized_image',
                                util.tensor2im(generated.data[0],imtype)),
                               ('real_image', 
                               util.tensor2im(data['image'][0],imtype))])
        if save_fake:
            visualizer.display_current_results(visuals_val, epoch, total_steps)

        gen_image = visuals_val['synthesized_image']
        gt_image = visuals_val['real_image']
        print('---------')
        print(generated.data[0].shape)
        print(gt_image.shape)
        print(gen_image.shape)
        # if opt.val_metric == "ssim":
        if gt_image.shape[2] == 3: # RGB
            tmp_ssim = [] 
            tmp_psnr = []
            for i in range(0,3):
                channel_score_ssim = ssim(gt_image[:,:,i], gen_image[:,:,i])
                tmp_ssim.append(channel_score_ssim)
                channel_score_psnr = psnr(gt_image[:,:,i], gen_image[:,:,i])
                tmp_psnr.append(channel_score_psnr)
            score_ssim = np.mean(tmp_ssim)
            score_psnr = np.mean(tmp_psnr)
            
        else:
            score_ssim = ssim(gt_image, gen_image)

        ssim_scores.append(score_ssim)
        psnr_scores.append(score_psnr)
        
    avg_ssim_val = np.mean(ssim_scores)
    print('Averagae SSIM for Epoch {} Val = {}'.format(epoch, avg_ssim_val))
    avg_psnr_val = np.mean(psnr_scores)
    print('Averagae PSNR for Epoch {} Val = {}'.format(epoch, avg_psnr_val))

    ##Save scores

    if epoch == 1:
        file_name = os.path.join(opt.checkpoints_dir, opt.name, 'train_val_scores.json')
        first_entry = [{'epoch': epoch,
                        'dataset': 'train',
                        'ssim': avg_ssim_train,
                        'psnr': avg_psnr_train}]
        second_entry = {'epoch': epoch,
                        'dataset': 'val',
                        'ssim': avg_ssim_val,
                        'psnr': avg_psnr_val}

        with open(file_name, mode='w') as f:
            json.dump(first_entry, f)
        with open(file_name, mode='r') as f:
            data = json.load(f)
            data.append(second_entry)
        with open(file_name, 'w') as f:
            json.dump(data,f)
    else:
        train_entry = {'epoch': epoch,
                        'dataset': 'train',
                        'ssim': avg_ssim_train,
                        'psnr': avg_psnr_train}
        val_entry = {'epoch': epoch,
                         'dataset': 'val',
                         'ssim': avg_ssim_val,
                         'psnr': avg_psnr_val}

        with open(file_name, 'r') as f:
            data = json.load(f)
            data.append(train_entry)
            data.append(val_entry)
        with open(file_name, 'w') as f:
            json.dump(data,f)

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
