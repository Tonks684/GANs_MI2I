import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    This class defines the training options for the pix2pixHD model.

    Args:
        BaseOptions (class): The base options class.

    Attributes:
        display_freq (int): Frequency of showing training results on screen.
        print_freq (int): Frequency of showing training results on console.
        save_latest_freq (int): Frequency of saving the latest results.
        save_epoch_freq (int): Frequency of saving checkpoints at the end of epochs.
        no_html (bool): Flag to indicate whether to save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/.
        debug (bool): Flag to indicate whether to only do one epoch and display at each iteration.
        continue_train (bool): Flag to indicate whether to continue training by loading the latest model.
        load_pretrain (str): Path to the pretrained model to load from.
        which_epoch (str): Which epoch to load? Set to 'latest' to use the latest cached model.
        phase (str): Phase of the training process ('train', 'val', 'test', etc).
        niter (int): Number of iterations at starting learning rate.
        niter_decay (int): Number of iterations to linearly decay learning rate to zero.
        beta1 (float): Momentum term of Adam optimizer.
        lr (float): Initial learning rate for Adam optimizer.
        num_D (int): Number of discriminators to use.
        n_layers_D (int): Only used if which_model_netD==n_layers.
        ndf (int): Number of filters in the first convolutional layer of the discriminator.
        lambda_feat (float): Weight for feature matching loss.
        no_ganFeat_loss (bool): If specified, do not use discriminator feature matching loss.
        no_vgg_loss (bool): If specified, do not use VGG feature matching loss.
        no_lsgan (bool): If specified, do not use least square GAN. If False, use vanilla GAN.
        pool_size (int): The size of the image buffer that stores previously generated images.
        isTrain (bool): Flag to indicate whether the options are for training.

    """
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--n_epochs', type=int, default='2', help='number_epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        
        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss', default=True)        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        
        self.isTrain = True
