import time
import os
from tifffile import imsave, imread
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
from data.data_loader_dlmbl import CreateDataLoader    
import util.util as util
from util.visualizer import Visualizer
from pathlib import Path
from tensorboardX import SummaryWriter
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0


def train_epoch(opt, model, visualizer, dataset_train, optimizer_G, optimizer_D, total_steps, epoch, epoch_iter, display_delta):
    """
    Train the model for one epoch.

    Args:
        opt (argparse.Namespace): The command line arguments.
        model: The model to be trained.
        visualizer: The visualizer object for displaying and plotting errors.
        dataset_train: The training dataset.
        optimizer_G: The optimizer for the generator.
        optimizer_D: The optimizer for the discriminator.
        total_steps (int): The total number of training steps.
        epoch (int): The current epoch number.
        epoch_iter (int): The current iteration within the epoch.
        display_delta (int): The interval for displaying output images.
    

    Returns:
        tuple: A tuple containing the average losses and metrics for the epoch.
            - running_loss_D_fake (float): The average loss of the discriminator on fake samples.
            - running_loss_D_real (float): The average loss of the discriminator on real samples.
            - running_loss_G_GAN (float): The average GAN loss of the generator.
            - running_loss_G_GAN_Feat (float): The average feature matching loss of the generator.
            - running_loss_G_VGG (float): The average VGG loss of the generator.
            - mean_ssim_scores (float): The average SSIM score of the generated images.
            - mean_psnr_scores (float): The average PSNR score of the generated images.
    """
    
    running_loss_G_GAN = 0
    running_loss_D_real = 0
    running_loss_D_fake = 0
    running_loss_G_VGG = 0
    running_loss_G_GAN_Feat = 0
    ssim_scores = []
    psnr_scores = [] 
    dataset_size = len(dataset_train)   
    display_delta = total_steps % opt.display_freq #
    print_delta = total_steps % opt.print_freq

    for i, data in enumerate(dataset_train, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']),Variable(data['image']), Variable(data['feat']), infer=True)
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))
        # calculate final loss scalar
        loss_D_fake = loss_dict['D_fake']
        loss_D_real = loss_dict['D_real']
        loss_D =(loss_D_fake+loss_D_real) * 0.5
        
        loss_G_GAN = loss_dict['G_GAN'] 
        loss_G_GAN_Feat = loss_dict.get('G_GAN_Feat',0) 
        loss_G_VGG = loss_dict.get('G_VGG',0)
        loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG
        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            from apex import amp
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
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps) 
        ### display output images
        visuals = OrderedDict([('label',
                                util.tensor2im(data['label'][0],imtype=np.uint16)),
                               ('synthesized_image',
                                util.tensor2im(generated.data[0],imtype=np.uint16)),
                               ('real_image', util.tensor2im(data['image'][0],imtype=np.uint16))])
        if save_fake:
            visualizer.display_current_results(visuals, epoch, total_steps)
        
        # Compute metric
        gen_image = visuals['synthesized_image'][:,:,0]
        gt_image = visuals['real_image'][:,:,0]
        score_ssim = ssim(gt_image, gen_image)
        ssim_scores.append(score_ssim)
        score_psnr = psnr(gt_image, gen_image)
        psnr_scores.append(score_psnr)

        if epoch_iter >= dataset_size:
                break
        running_loss_G_GAN += loss_G_GAN
        running_loss_G_GAN_Feat += loss_G_GAN_Feat
        running_loss_D_real += loss_D_real
        running_loss_D_fake += loss_D_fake
        running_loss_G_VGG += loss_G_VGG
    return running_loss_D_fake / dataset_size, running_loss_D_real/ dataset_size, running_loss_G_GAN / dataset_size, running_loss_G_GAN_Feat / dataset_size, running_loss_G_VGG/ dataset_size, np.mean(ssim_scores), np.mean(psnr_scores)

def val_epoch(opt, model, dataset_val, epoch):
    """
    Perform validation for one epoch.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataset_val (torch.utils.data.Dataset): The validation dataset.
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple containing the following:
            - list: A list of loss values and evaluation metrics.
            - list: A list of generated images.
            - list: A list of real images.
            - dict: The label data.

    """
    ssim_scores = []
    psnr_scores = []
    running_loss_G_GAN = 0
    running_loss_D_real = 0
    running_loss_D_fake = 0
    running_loss_G_VGG = 0
    running_loss_G_GAN_Feat = 0
    model.eval()
    with torch.no_grad():
        for data in dataset_val:
            print(data['label'].shape)
            losses, generated = model(Variable(data['label']), Variable(data['inst']),Variable(data['image']), Variable(data['feat']), infer=True)
            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))
            # calculate final loss scalar
            losses, generated = model(Variable(data['label']), Variable(data['inst']),Variable(data['image']), Variable(data['feat']), infer=True)
            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))
            # calculate final loss scalar
            loss_D_fake = loss_dict['D_fake'] * 0.5
            loss_D_real = loss_dict['D_real'] * 0.5
            
            loss_G_GAN = loss_dict['G_GAN'] 
            loss_G_GAN_Feat = loss_dict.get('G_GAN_Feat',0) 
            loss_G_VGG = loss_dict.get('G_VGG',0)
            running_loss_D_fake += loss_D_fake
            running_loss_D_real += loss_D_real
            running_loss_G_GAN += loss_G_GAN
            running_loss_G_GAN_Feat += loss_G_GAN_Feat
            running_loss_G_VGG += loss_G_VGG
            
            visuals = OrderedDict([('input_label',
                                util.tensor2im(data['label'][0],imtype=np.float32)),
                               ('synthesized_image',
                                util.tensor2im(generated.data[0],imtype='dlmbl')),
                               ('real_image', util.tensor2im(data['image'][0],imtype='dlmbl'))])
            gen_image = visuals['synthesized_image'][:,:,0]
            gt_image = visuals['real_image'][:,:,0]
            
            score_ssim = ssim(gt_image, gen_image)
            ssim_scores.append(score_ssim)
            
            score_psnr = psnr(gt_image, gen_image)
            psnr_scores.append(score_psnr)
        
        return [running_loss_D_fake / len(dataset_val), running_loss_D_real/ len(dataset_val), running_loss_G_GAN / len(dataset_val), running_loss_G_GAN_Feat / len(dataset_val), running_loss_G_VGG/ len(dataset_val), np.mean(ssim_scores), np.mean(psnr_scores)],  util.tensors2ims(opt, generated.data,imtype='dlmbl'), util.tensors2ims(opt,data['image'],imtype='dlmbl'), data['label']



def train(opt, model, visualizer, dataset_train, dataset_val, optimizer_G, optimizer_D, start_epoch, epoch_iter, iter_path, display_delta, writer):
    """
    Trains the model using the specified options and datasets.

    Args:
        opt (argparse.Namespace): The options for training.
        model: The model to be trained.
        visualizer: The visualizer for displaying training progress.
        dataset_train: The training dataset.
        dataset_val: The validation dataset.
        optimizer_G: The optimizer for the generator.
        optimizer_D: The optimizer for the discriminator.
        start_epoch (int): The starting epoch for training.
        epoch_iter (int): The current iteration within the epoch.
        iter_path: The path to save the current iteration.
        writer: The writer for logging training metrics.

    Returns:
        None - but training outputs are saved to Tensorboard
    """
    total_steps = (start_epoch-1) * (len(dataset_train)+len(dataset_val)) + epoch_iter 
    for epoch in range(start_epoch, opt.n_epochs):
        epoch_start_time = time.time()
        if epoch == start_epoch:
            dummy_input = (torch.rand(1,1,opt.loadSize,opt.loadSize),torch.rand(1,1,opt.loadSize,opt.loadSize),torch.rand(1,1,opt.loadSize,opt.loadSize),torch.rand(1,1,opt.loadSize,opt.loadSize))
            writer.add_graph(model.module.netG, dummy_input)    
        else:
            epoch_iter = epoch_iter % len(dataset_train)

        train_loss_D_fake, train_loss_D_real, train_loss_G_GAN, train_loss_G_Feat, train_loss_G_VGG, mean_ssim, mean_psnr = train_epoch(opt, model, visualizer, dataset_train, optimizer_G, optimizer_D, total_steps, epoch, epoch_iter, iter_path, display_delta)
        
        [val_loss_D_fake, val_loss_D_real, val_loss_G_GAN, val_loss_G_Feat, val_loss_G_VGG, val_ssim, val_psnr], virtual_stain, fluorescence, brightfield = val_epoch(model, dataset_val, epoch)
        
        visualizer.results_plot(brightfield,fluorescence,virtual_stain,['Phase Contrast', 'Fluorescence', 'Virtual Stain'],writer,epoch,rows=brightfield.shape[0])

        # Tensorboard Logging
        epoch_discriminator = {'fake_is_fake': train_loss_D_fake, 'real_is_real': train_loss_D_real}
        writer.add_scalars('Discriminator Predicted Probability on Training Set', epoch_discriminator, epoch)
        
        epoch_discriminator = {'fake_is_fake': val_loss_D_fake, 'real_is_real': val_loss_D_real}
        writer.add_scalars('Discriminator Predicted Probability on Validation Set', epoch_discriminator, epoch)
        
        epoch_generator = {'train': train_loss_G_GAN, 'validation': val_loss_G_GAN}
        writer.add_scalars('Generator Least Square Loss', epoch_generator, epoch)
        
        epoch_generator = {'train': train_loss_G_Feat, 'validation': val_loss_G_Feat}
        writer.add_scalars('Generator Feature Matching Loss', epoch_generator, epoch)

        epoch_ssim = {'train': mean_ssim, 'validation': val_ssim}
        writer.add_scalars('SSIM', epoch_ssim, epoch)
        epoch_psnr = {'train': mean_psnr, 'validation': val_psnr}
        writer.add_scalars('PSNR', epoch_psnr, epoch)

    print('Training Losses: D_fake: {}, D_real: {}, G_GAN: {}, G_GAN_Feat: {}, G_VGG: {}'.format(train_loss_D_fake, train_loss_D_real, train_loss_G_GAN, train_loss_G_Feat, train_loss_G_VGG))
    print('Validation Losses: D_fake: {}, D_real: {}, G_GAN: {}, G_GAN_Feat: {}, G_VGG: {}'.format(val_loss_D_fake, val_loss_D_real, val_loss_G_GAN, val_loss_G_Feat, val_loss_G_VGG))
    print('SSIM: Train: {}, Validation: {}'.format(mean_ssim, val_ssim))
    print('PSNR: Train: {}, Validation: {}'.format(mean_psnr, val_psnr))    
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
    