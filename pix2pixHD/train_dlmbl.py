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
import util.my_util as util
from util.my_visualizer import Visualizer
from pathlib import Path
from tensorboardX import SummaryWriter
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0


def train_epoch(opt, model, visualizer, dataset_train, dataset_size, optimizer_G, optimizer_D, total_steps, epoch, epoch_iter, iter_path, display_delta, print_delta, save_delta):
    running_loss_G_GAN = 0
    running_loss_D_real = 0
    running_loss_D_fake = 0
    running_loss_G_VGG = 0
    running_loss_G_GAN_Feat = 0
    ssim_scores = []
    psnr_scores = []    
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
    return running_loss_D_fake / len(dataset_train), running_loss_D_real/ len(dataset_train), running_loss_G_GAN / len(dataset_train), running_loss_G_GAN_Feat / len(dataset_train), running_loss_G_VGG/ len(dataset_train), np.mean(ssim_scores), np.mean(psnr_scores)

def val_epoch(model, dataset_val, epoch):
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
                                util.tensor2im(data['label'][0],imtype=np.uint16)),
                               ('synthesized_image',
                                util.tensor2im(generated.data[0],imtype=np.uint16)),
                               ('real_image', util.tensor2im(data['image'][0],imtype=np.uint16))])
            gen_image = visuals['synthesized_image'][:,:,0]
            gt_image = visuals['real_image'][:,:,0]
            
            score_ssim = ssim(gt_image, gen_image)
            ssim_scores.append(score_ssim)
            
            score_psnr = psnr(gt_image, gen_image)
            psnr_scores.append(score_psnr)
            bf = data['label'].cpu().detach().numpy()
            fl = data['image'].cpu().detach().numpy()
            vs = generated.data.cpu().detach().numpy() 

            imsave(f'{opt.checkpoints_dir}/bf/epoch_{epoch}.tiff',bf.astype(np.float32),imagej=True)
            imsave(f'{opt.checkpoints_dir}/fl/epoch_{epoch}.tiff',fl.astype(np.float32),imagej=True)
            imsave(f'{opt.checkpoints_dir}/vs/epoch_{epoch}.tiff',vs.astype(np.float32),imagej=True)
        return [running_loss_D_fake / len(dataset_val), running_loss_D_real/ len(dataset_val), running_loss_G_GAN / len(dataset_val), running_loss_G_GAN_Feat / len(dataset_val), running_loss_G_VGG/ len(dataset_val), np.mean(ssim_scores), np.mean(psnr_scores)],  util.tensors2ims(generated.data,imtype=np.uint16), util.tensors2ims(data['image'],imtype=np.uint16), data['label']



def train(opt, model, visualizer, dataset_train, dataset_size, optimizer_G, optimizer_D, total_steps, start_epoch, epoch_iter, iter_path, display_delta, print_delta, save_delta,writer):
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        train_loss_D_fake, train_loss_D_real, train_loss_G_GAN, train_loss_G_Feat, train_loss_G_VGG, mean_ssim, mean_psnr = train_epoch(opt, model, visualizer, dataset_train, dataset_size, optimizer_G, optimizer_D, total_steps, epoch, epoch_iter, iter_path, display_delta, print_delta, save_delta)
        [val_loss_D_fake, val_loss_D_real, val_loss_G_GAN, val_loss_G_Feat, val_loss_G_VGG, val_ssim, val_psnr], virtual_stain, fluorescence, brightfield = val_epoch(model, dataset_val,epoch)
        visualizer.results_plot(brightfield,fluorescence,virtual_stain,['Bright-field', 'Fluorescence', 'Virtual Stain'],writer,epoch,rows=brightfield.shape[0])

    # Tensorboard Logging
        epoch_discriminator = {'D_fake': train_loss_D_fake, 'D_real': train_loss_D_real}
        writer.add_scalars('Discriminator Probabilities Train', epoch_discriminator, epoch)
        epoch_discriminator = {'D_fake': val_loss_D_fake, 'D_real': val_loss_D_real}
        writer.add_scalars('Discriminator Probabilities Validation', epoch_discriminator, epoch)
        epoch_generator = {'G_GAN_Train': train_loss_G_GAN, 'G_GAN_Validation': val_loss_G_GAN}
        writer.add_scalars('Generator Loss GAN', epoch_generator, epoch)
        epoch_generator = {'G_GAN_Feat_Train': train_loss_G_Feat, 'G_GAN_Feat_Validation': val_loss_G_Feat}
        writer.add_scalars('Generator Loss GAN Feat', epoch_generator, epoch)
        epoch_generator = {'G_VGG_Train': train_loss_G_VGG, 'G_VGG_Validation': val_loss_G_VGG}
        writer.add_scalars('Generator Loss VGG', epoch_generator, epoch)
        epoch_ssim = {'SSIM_Train': mean_ssim, 'SSIM_Validation': val_ssim}
        writer.add_scalars('SSIM', epoch_ssim, epoch)
        epoch_psnr = {'PSNR_Train': mean_psnr, 'PSNR_Validation': val_psnr}
        writer.add_scalars('PSNR', epoch_psnr, epoch)


        

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
    print('Training Losses: D_fake: {}, D_real: {}, G_GAN: {}, G_GAN_Feat: {}, G_VGG: {}'.format(train_loss_D_fake, train_loss_D_real, train_loss_G_GAN, train_loss_G_Feat, train_loss_G_VGG))
    print('Validation Losses: D_fake: {}, D_real: {}, G_GAN: {}, G_GAN_Feat: {}, G_VGG: {}'.format(val_loss_D_fake, val_loss_D_real, val_loss_G_GAN, val_loss_G_Feat, val_loss_G_VGG))

if __name__ == '__main__':
    opt = TrainOptions().parse()
    writer = SummaryWriter(opt.checkpoints_dir)
    
    util.set_seed(int(opt.seed))
    
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
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
    Path(f'{opt.checkpoints_dir}/bf/').mkdir(parents=True, exist_ok=True)
    Path(f'{opt.checkpoints_dir}/vs/').mkdir(parents=True, exist_ok=True)
    Path(f'{opt.checkpoints_dir}/fl/').mkdir(parents=True, exist_ok=True)
    
    if opt.fp16:
        from apex import amp
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    print("Total Steps {}".format(total_steps))
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
   
    train(opt, model, visualizer, dataset_train, dataset_size, optimizer_G, optimizer_D, total_steps, start_epoch, epoch_iter, iter_path, display_delta, print_delta, save_delta, writer)
    iter_end_time = time.time()
    
