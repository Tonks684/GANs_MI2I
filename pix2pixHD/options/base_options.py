#pix2pixHD/options/base_options.py
import argparse
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from util import util
import torch


class BaseOptions:
    """
    Base options class for the experiment.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        """
        Initialize the command line arguments.
        """
        # Experiment specifics
        self.parser.add_argument(
            "--name",
            type=str,
            default="label2city",
            help="Name of the experiment. It decides where to store samples and models",
        )
        self.parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="GPU ids: e.g. 0  0,1,2, 0,2. Use -1 for CPU",
        )
        self.parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./training/",
            help="Models are saved here",
        )
        self.parser.add_argument(
            "--model", type=str, default="pix2pixHD", help="Which model to use"
        )
        self.parser.add_argument(
            "--norm",
            type=str,
            default="instance",
            help="Instance normalization or batch normalization",
        )
        self.parser.add_argument(
            "--use_dropout", action="store_true", help="Use dropout for the generator"
        )
        self.parser.add_argument(
            "--data_type",
            default=32,
            type=int,
            choices=[8, 16, 32, 64],
            help="Supported data type i.e. 8, 16, 32, 64 bit",
        )
        self.parser.add_argument(
            "--input_is_float", action="store_true", help="Is input float or integer"
        )  # Changed from --is_float to --input_is_float
        self.parser.add_argument(
            "--verbose", action="store_true", default=False, help="Toggles verbose"
        )
        self.parser.add_argument(
            "--fp16", action="store_true", default=False, help="Train with AMP"
        )  # Changed from --fp16 to --use_fp16
        self.parser.add_argument(
            "--local_rank",
            type=int,
            default=0,
            help="Local rank for distributed training",
        )

        # Input/output sizes
        self.parser.add_argument(
            "--batchSize", type=int, default=1, help="Input batch size"
        )
        self.parser.add_argument(
            "--loadSize", type=int, default=1024, help="Scale images to this size"
        )
        self.parser.add_argument(
            "--fineSize", type=int, default=512, help="Then crop to this size"
        )
        self.parser.add_argument(
            "--label_nc", type=int, default=0, help="# of input label channels"
        )
        self.parser.add_argument(
            "--input_nc", type=int, default=3, help="# of input image channels"
        )
        self.parser.add_argument(
            "--output_nc", type=int, default=3, help="# of output image channels"
        )
        self.parser.add_argument(
            "--input_RGB", action="store_true", help="Is input RGB?"
        )
        self.parser.add_argument(
            "--output_RGB", action="store_true", help="Is output RGB?"
        )
        self.parser.add_argument("--seed", type=int, help="Set seed for run")

        # For setting inputs
        self.parser.add_argument(
            "--dataroot", type=str, default="./datasets/cityscapes/"
        )
        self.parser.add_argument(
            "--target",
            type=str,
            default="nuclei",
            help="Folder name containing target virtual stain for training",
        )
        self.parser.add_argument(
            "--resize_or_crop",
            type=str,
            default="scale_width",
            help="Scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]",
        )
        self.parser.add_argument(
            "--serial_batches",
            action="store_true",
            help="If true, takes images in order to make batches, otherwise takes them randomly",
        )
        self.parser.add_argument(
            "--no_flip",
            action="store_true",
            default=True,
            help="If specified, do not flip the images for data augmentation",
        )
        self.parser.add_argument(
            "--nThreads", type=int, default=2, help="# threads for loading data"
        )
        self.parser.add_argument(
            "--max_dataset_size",
            type=int,
            default=float("inf"),
            help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
        )

        # For displays
        self.parser.add_argument(
            "--display_winsize", type=int, default=512, help="Display window size"
        )
        self.parser.add_argument(
            "--tf_log",
            action="store_true",
            help="If specified, use TensorBoard logging. Requires TensorFlow installed",
        )

        # For generator
        self.parser.add_argument(
            "--netG", type=str, default="global", help="Selects model to use for netG"
        )
        self.parser.add_argument(
            "--ngf", type=int, default=64, help="# of gen filters in first conv layer"
        )
        self.parser.add_argument(
            "--n_downsample_global",
            type=int,
            default=4,
            help="Number of downsampling layers in netG",
        )
        self.parser.add_argument(
            "--n_blocks_global",
            type=int,
            default=9,
            help="Number of residual blocks in the global generator network",
        )
        self.parser.add_argument(
            "--n_blocks_local",
            type=int,
            default=3,
            help="Number of residual blocks in the local enhancer network",
        )
        self.parser.add_argument(
            "--n_local_enhancers",
            type=int,
            default=1,
            help="Number of local enhancers to use",
        )
        self.parser.add_argument(
            "--niter_fix_global",
            type=int,
            default=0,
            help="Number of epochs that we only train the outmost local enhancer",
        )
        self.parser.add_argument(
            "--dropout_variation_inf",
            choices=[True, False],
            default=False,
            type=bool,
            help="If True, turning dropout of 0.2 on for variation inference",
        )

        # For instance-wise features
        self.parser.add_argument(
            "--no_instance",
            action="store_true",
            default=True,
            help="If specified, do *not* add instance map as input",
        )
        self.parser.add_argument(
            "--instance_feat",
            action="store_true",
            help="If specified, add encoded instance features as input",
        )
        self.parser.add_argument(
            "--label_feat",
            action="store_true",
            help="If specified, add encoded label features as input",
        )
        self.parser.add_argument(
            "--feat_num", type=int, default=3, help="Vector length for encoded features"
        )
        self.parser.add_argument(
            "--load_features",
            action="store_true",
            help="If specified, load precomputed feature maps",
        )
        self.parser.add_argument(
            "--n_downsample_E",
            type=int,
            default=4,
            help="# of downsampling layers in encoder",
        )
        self.parser.add_argument(
            "--nef",
            type=int,
            default=16,
            help="# of encoder filters in the first conv layer",
        )
        self.parser.add_argument(
            "--n_clusters", type=int, default=10, help="Number of clusters for features"
        )
        self.parser.add_argument(
            "--output_reshape",
            type=int,
            help="Resize model output to this shape, fixed to same for x and y",
        )

        self.initialized = True

if __name__ == "__main__":
    opt = BaseOptions().parse()
    print(opt)
