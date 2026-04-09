import logging
import math
import os
import time

import numpy as np
import torch
import util.util as util
from data.data_loader_dlmbl import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter
from util.visualizer import Visualizer

logger = logging.getLogger(__name__)

# W&B is optional — import lazily so the codebase works without it installed
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def train_epoch(opt, model, visualizer, dataset_train, optimizer_G, optimizer_D,
                total_steps, epoch, epoch_iter, scaler_G=None, scaler_D=None):
    """
    Train the model for one epoch.

    Args:
        opt: Training options namespace.
        model: The pix2pixHD model (DataParallel-wrapped).
        visualizer: Visualizer for logging errors.
        dataset_train: Training DataLoader.
        optimizer_G: Generator optimiser.
        optimizer_D: Discriminator optimiser.
        total_steps (int): Global step counter.
        epoch (int): Current epoch number.
        epoch_iter (int): Iteration offset within the epoch.
        scaler_G: torch.cuda.amp.GradScaler for the generator (or None).
        scaler_D: torch.cuda.amp.GradScaler for the discriminator (or None).

    Returns:
        Tuple of per-epoch average losses:
        (D_fake, D_real, G_GAN, G_GAN_Feat, G_VGG)
    """
    model.train()
    running_loss_G_GAN = 0
    running_loss_D_real = 0
    running_loss_D_fake = 0
    running_loss_G_VGG = 0
    running_loss_G_GAN_Feat = 0
    dataset_size = len(dataset_train)
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq

    for i, data in enumerate(dataset_train, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            losses, generated = model(
                data['label'], data['inst'],
                data['image'], data['feat'], infer=save_fake,
            )

        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        loss_D_fake = loss_dict['D_fake'] * 0.5
        loss_D_real = loss_dict['D_real'] * 0.5
        loss_D = loss_D_fake + loss_D_real

        loss_G_GAN = loss_dict['G_GAN']
        loss_G_GAN_Feat = loss_dict.get('G_GAN_Feat', 0)
        loss_G_VGG = loss_dict.get('G_VGG', 0)
        loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG

        ############### Backward Pass ####################
        optimizer_G.zero_grad(set_to_none=True)
        if scaler_G is not None:
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
        else:
            loss_G.backward()
            optimizer_G.step()

        optimizer_D.zero_grad(set_to_none=True)
        if scaler_D is not None:
            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
        else:
            loss_D.backward()
            optimizer_D.step()

        errors = {k: v.data.item() if not isinstance(v, int) else v
                  for k, v in loss_dict.items()}
        t = (time.time() - iter_start_time) / opt.print_freq
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        running_loss_G_GAN += loss_G_GAN
        running_loss_G_GAN_Feat += loss_G_GAN_Feat
        running_loss_D_real += loss_D_real
        running_loss_D_fake += loss_D_fake
        running_loss_G_VGG += loss_G_VGG

    return (running_loss_D_fake / dataset_size, running_loss_D_real / dataset_size,
            running_loss_G_GAN / dataset_size, running_loss_G_GAN_Feat / dataset_size,
            running_loss_G_VGG / dataset_size)


def val_epoch(opt, model, dataset_val):
    """
    Perform validation for one epoch.

    Args:
        opt: Training options namespace.
        model: The pix2pixHD model.
        dataset_val: Validation DataLoader.

    Returns:
        Tuple of (losses_list, virtual_stain, fluorescence, input_data)
    """
    model.eval()
    running_loss_G_GAN = 0
    running_loss_D_real = 0
    running_loss_D_fake = 0
    running_loss_G_VGG = 0
    running_loss_G_GAN_Feat = 0

    with torch.no_grad():
        for data in dataset_val:
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                losses, generated = model(
                    data['label'], data['inst'],
                    data['image'], data['feat'], infer=True,
                )
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            loss_D_fake = loss_dict['D_fake'] * 0.5
            loss_D_real = loss_dict['D_real'] * 0.5
            loss_G_GAN = loss_dict['G_GAN']
            loss_G_GAN_Feat = loss_dict.get('G_GAN_Feat', 0)
            loss_G_VGG = loss_dict.get('G_VGG', 0)

            running_loss_D_fake += loss_D_fake
            running_loss_D_real += loss_D_real
            running_loss_G_GAN += loss_G_GAN
            running_loss_G_GAN_Feat += loss_G_GAN_Feat
            running_loss_G_VGG += loss_G_VGG

    # Use last batch for visualisation
    input_data = util.tensors2ims(opt, data['label'], imtype=np.float32)
    virtual_stain = util.tensors2ims(opt, generated.data, imtype='dlmbl')
    fluorescence = util.tensors2ims(opt, data['image'], imtype='dlmbl')

    n = len(dataset_val)
    return (
        [running_loss_D_fake / n, running_loss_D_real / n,
         running_loss_G_GAN / n, running_loss_G_GAN_Feat / n,
         running_loss_G_VGG / n],
        virtual_stain,
        fluorescence,
        input_data,
    )


def train(opt, model, visualizer, dataset_train, dataset_val,
          optimizer_G, optimizer_D, start_epoch, epoch_iter, writer):
    """
    Train the model for all epochs, logging to TensorBoard and optionally W&B.

    Args:
        opt: Training options namespace.
        model: The pix2pixHD model.
        visualizer: Visualizer instance.
        dataset_train: Training DataLoader.
        dataset_val: Validation DataLoader.
        optimizer_G: Generator optimiser.
        optimizer_D: Discriminator optimiser.
        start_epoch (int): First epoch (for resuming).
        epoch_iter (int): Iteration offset within start_epoch.
        writer: TensorBoard SummaryWriter.
    """
    use_wandb = getattr(opt, 'use_wandb', False) and _WANDB_AVAILABLE

    # AMP scalers (only active when fp16=True)
    scaler_G = torch.cuda.amp.GradScaler() if opt.fp16 else None
    scaler_D = torch.cuda.amp.GradScaler() if opt.fp16 else None

    total_steps = (start_epoch - 1) * (len(dataset_train) + len(dataset_val)) + epoch_iter
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    for epoch in range(start_epoch, opt.n_epochs):
        epoch_start_time = time.time()
        epoch_iter = epoch_iter % len(dataset_train)

        train_losses = train_epoch(
            opt, model, visualizer, dataset_train,
            optimizer_G, optimizer_D, total_steps, epoch, epoch_iter,
            scaler_G=scaler_G, scaler_D=scaler_D,
        )
        train_loss_D_fake, train_loss_D_real, train_loss_G_GAN, train_loss_G_Feat, train_loss_G_VGG = train_losses

        val_losses, virtual_stain, fluorescence, brightfield = val_epoch(opt, model, dataset_val)
        val_loss_D_fake, val_loss_D_real, val_loss_G_GAN, val_loss_G_Feat, val_loss_G_VGG = val_losses

        visualizer.results_plot(
            brightfield, fluorescence, virtual_stain,
            ['Phase Contrast', 'Fluorescence', 'Virtual Stain'],
            writer, epoch, rows=brightfield.shape[0],
        )

        # TensorBoard logging
        writer.add_scalars('Discriminator/Train', {
            'fake_is_fake': train_loss_D_fake, 'real_is_real': train_loss_D_real,
        }, epoch)
        writer.add_scalars('Discriminator/Val', {
            'fake_is_fake': val_loss_D_fake, 'real_is_real': val_loss_D_real,
        }, epoch)
        writer.add_scalars('Generator/GAN_Loss', {
            'train': train_loss_G_GAN, 'val': val_loss_G_GAN,
        }, epoch)
        writer.add_scalars('Generator/Feature_Matching_Loss', {
            'train': train_loss_G_Feat, 'val': val_loss_G_Feat,
        }, epoch)

        # W&B logging
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/D_fake': train_loss_D_fake,
                'train/D_real': train_loss_D_real,
                'train/G_GAN': train_loss_G_GAN,
                'train/G_Feat': train_loss_G_Feat,
                'train/G_VGG': train_loss_G_VGG,
                'val/D_fake': val_loss_D_fake,
                'val/D_real': val_loss_D_real,
                'val/G_GAN': val_loss_G_GAN,
                'val/G_Feat': val_loss_G_Feat,
                'val/G_VGG': val_loss_G_VGG,
                'epoch_time_s': int(time.time() - epoch_start_time),
            }
            # Log a grid of sample images every save_epoch_freq epochs
            if epoch % opt.save_epoch_freq == 0:
                def _to_wandb_image(arr, caption):
                    # arr may be (H,W) or (H,W,C); normalise to uint8 for W&B
                    a = arr.squeeze()
                    a = ((a - a.min()) / (a.ptp() + 1e-9) * 255).astype(np.uint8)
                    return wandb.Image(a, caption=caption)

                log_dict['val/phase_contrast'] = _to_wandb_image(brightfield[0], 'Phase Contrast')
                log_dict['val/virtual_stain'] = _to_wandb_image(virtual_stain[0], 'Virtual Stain')
                log_dict['val/fluorescence'] = _to_wandb_image(fluorescence[0], 'GT Fluorescence')
            wandb.log(log_dict, step=epoch)

        logger.info(
            'Epoch %d/%d  Train D_fake=%.4f D_real=%.4f G_GAN=%.4f G_Feat=%.4f G_VGG=%.4f',
            epoch, opt.n_epochs,
            train_loss_D_fake, train_loss_D_real,
            train_loss_G_GAN, train_loss_G_Feat, train_loss_G_VGG,
        )
        logger.info(
            'Epoch %d/%d  Val   D_fake=%.4f D_real=%.4f G_GAN=%.4f G_Feat=%.4f G_VGG=%.4f  time=%ds',
            epoch, opt.n_epochs,
            val_loss_D_fake, val_loss_D_real,
            val_loss_G_GAN, val_loss_G_Feat, val_loss_G_VGG,
            int(time.time() - epoch_start_time),
        )

        if epoch % opt.save_epoch_freq == 0:
            logger.info('Saving model at end of epoch %d, iter %d', epoch, total_steps)
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        if opt.niter_fix_global != 0 and epoch == opt.niter_fix_global:
            model.module.update_fixed_params()

        if epoch > opt.niter:
            model.module.update_learning_rate()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    opt = TrainOptions().parse()

    # Weights & Biases — initialise if requested
    if getattr(opt, 'use_wandb', False):
        if not _WANDB_AVAILABLE:
            logger.warning('--use_wandb specified but wandb is not installed. '
                           'Run: pip install wandb')
        else:
            wandb.init(
                project=opt.wandb_project,
                name=opt.wandb_run_name or opt.name,
                config=vars(opt),
                resume='allow',
            )
            logger.info('W&B run: %s', wandb.run.url)

    # Reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True finds the fastest convolution algorithm for fixed input sizes
        torch.backends.cudnn.benchmark = True

    # Data loaders
    data_loader_train = CreateDataLoader(opt)
    dataset_train = data_loader_train.load_data()
    data_loader_val = CreateDataLoader(opt, phase='val')
    dataset_val = data_loader_val.load_data()
    logger.info('Train size: %d  Val size: %d', len(data_loader_train), len(data_loader_val))

    # Model
    model = create_model(opt)

    # Compile generator and discriminator for a free ~10–30% throughput gain
    # (requires PyTorch >= 2.0; falls back gracefully on older versions)
    if hasattr(torch, 'compile'):
        try:
            model.module.netG = torch.compile(model.module.netG)
            model.module.netD = torch.compile(model.module.netD)
            logger.info('torch.compile applied to netG and netD')
        except Exception as e:
            logger.warning('torch.compile failed, continuing without it: %s', e)

    visualizer = Visualizer(opt)
    writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard'))

    # Resume support
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except FileNotFoundError:
            start_epoch, epoch_iter = 1, 0
    else:
        start_epoch, epoch_iter = 1, 0
    logger.info('Starting from epoch %d, iter %d', start_epoch, epoch_iter)

    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    train(opt, model, visualizer, dataset_train, dataset_val,
          optimizer_G, optimizer_D, start_epoch, epoch_iter, writer)

    writer.close()
    if getattr(opt, 'use_wandb', False) and _WANDB_AVAILABLE:
        wandb.finish()
