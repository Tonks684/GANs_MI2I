import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from .base_options import BaseOptions
import torch
from util import util


class TrainOptions(BaseOptions):
    """
    This class defines the training options for the pix2pixHD model.
    """

    def initialize(self):
        BaseOptions.initialize(self)
        # For displays
        self.parser.add_argument(
            "--display_freq",
            type=int,
            default=100,
            help="Frequency of showing training results on screen",
        )
        self.parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="Frequency of showing training results on console",
        )
        self.parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=1000,
            help="Frequency of saving the latest results",
        )
        self.parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=20,
            help="Frequency of saving checkpoints at the end of epochs",
        )
        self.parser.add_argument(
            "--no_html",
            action="store_true",
            help="Do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Only do one epoch and displays at each iteration",
        )

        # For training
        self.parser.add_argument(
            "--n_epochs", type=int, default=200, help="Number of epochs"
        )
        self.parser.add_argument(
            "--continue_train",
            action="store_true",
            help="Continue training: load the latest model",
        )
        self.parser.add_argument(
            "--load_pretrain",
            type=str,
            default="",
            help="Load the pretrained model from the specified location",
        )
        self.parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="Which epoch to load? Set to latest to use latest cached model",
        )
        self.parser.add_argument(
            "--phase", type=str, default="train", help="Phase: train, val, test, etc"
        )
        self.parser.add_argument(
            "--niter",
            type=int,
            default=100,
            help="# of iterations at starting learning rate",
        )
        self.parser.add_argument(
            "--niter_decay",
            type=int,
            default=100,
            help="# of iterations to linearly decay learning rate to zero",
        )
        self.parser.add_argument(
            "--beta1", type=float, default=0.5, help="Momentum term of adam"
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.0002, help="Initial learning rate for adam"
        )

        # For discriminators
        self.parser.add_argument(
            "--num_D", type=int, default=2, help="Number of discriminators to use"
        )
        self.parser.add_argument(
            "--n_layers_D",
            type=int,
            default=3,
            help="Only used if which_model_netD==n_layers",
        )
        self.parser.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="# of discriminator filters in first conv layer",
        )
        self.parser.add_argument(
            "--lambda_feat",
            type=float,
            default=10.0,
            help="Weight for feature matching loss",
        )
        self.parser.add_argument(
            "--no_ganFeat_loss",
            action="store_true",
            help="If specified, do *not* use discriminator feature matching loss",
        )
        self.parser.add_argument(
            "--no_vgg_loss",
            action="store_true",
            help="If specified, do *not* use VGG feature matching loss",
            default=True,
        )
        self.parser.add_argument(
            "--no_lsgan",
            action="store_true",
            help="Do *not* use least square GAN, if false, use vanilla GAN",
        )
        self.parser.add_argument(
            "--pool_size",
            type=int,
            default=0,
            help="The size of image buffer that stores previously generated images",
        )

        self.isTrain = True

    def parse(self, save=True):
        """
        Parse the command line arguments and return the options.

        Args:
            save (bool): Whether to save the options to a file.

        Returns:
            argparse.Namespace: The parsed command line arguments.
        """
        if not self.initialized:
            self.initialize()

        # Filter out the Jupyter specific arguments
        jupyter_args = [
            "--ip",
            "--stdin",
            "--control",
            "--hb",
            "--Session.signature_scheme",
            "--Session.key",
            "--shell",
            "--transport",
            "--iopub",
            "--f",
        ]
        filtered_argv = [
            arg for arg in sys.argv if not any(jarg in arg for jarg in jupyter_args)
        ]

        self.opt, unknown = self.parser.parse_known_args(filtered_argv)
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # Set GPU ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # Print options
        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # Save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not getattr(self.opt, "continue_train", False):
            file_name = os.path.join(expr_dir, "opt.txt")
            with open(file_name, "wt") as opt_file:
                opt_file.write("------------ Options -------------\n")
                for k, v in sorted(args.items()):
                    opt_file.write("%s: %s\n" % (str(k), str(v)))
                opt_file.write("-------------- End ----------------\n")
        return self.opt


# Example usage
if __name__ == "__main__":
    opt = TrainOptions().parse()
    print(opt)
