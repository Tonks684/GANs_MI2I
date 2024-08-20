import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from .base_options import BaseOptions

class TestOptions(BaseOptions):
     """
     This class defines the testing options for the pix2pixHD model.

     Args:
          BaseOptions (class): The base options class.

     Attributes:
          parser (ArgumentParser): The argument parser object.
          ntest (int): The number of test examples.
          results_dir (str): The directory to save the results.
          aspect_ratio (float): The aspect ratio of the result images.
          phase (str): The phase of the model (train, val, test, etc).
          which_epoch (str): The epoch to load the model from.
          how_many (int): The number of test images to run.
          cluster_path (str): The path for clustered results of encoded features.
          use_encoded_image (bool): If True, encode the real image to get the feature map.
          export_onnx (str): The file path to export the ONNX model.
          engine (str): The serialized TRT engine to run.
          onnx (str): The ONNX model to run via TRT.
          variational_inf_runs (int): The number of runs for variational_inference.py.
          variational_inf_path (str): The path to save variational inf outputs.
          isTrain (bool): Whether the model is for training or testing.

     """

     def initialize(self):
          BaseOptions.initialize(self)
          self.parser.add_argument('--ntest', type=int, default=float("inf"),
                                         help='# of test examples.')
          self.parser.add_argument('--results_dir', type=str,
                                         default='./results/',
                                         help='saves results here.')
          self.parser.add_argument('--aspect_ratio',
                                         type=float, default=1.0,
                                         help='aspect ratio of result images')
          self.parser.add_argument('--phase', type=str, default='test',
                                         help='train, val, test, etc')
          self.parser.add_argument('--which_epoch', type=str,
                                         default='latest',
                                         help='which epoch to load? set to latest'
                                               ' to use latest cached model')
          self.parser.add_argument('--how_many', type=int, default=50,
                                         help='how many test images to run')
          self.parser.add_argument('--cluster_path', type=str,
                                         default='features_clustered_010.npy',
                                         help='the path for clustered results of '
                                               'encoded features')
          self.parser.add_argument('--use_encoded_image', action='store_true',
                                         help='if specified, encode the real image to '
                                               'get the feature map')
          self.parser.add_argument("--export_onnx", type=str,
                                         help="export ONNX model to a given file")
          self.parser.add_argument("--engine", type=str,
                                         help="run serialized TRT engine")
          self.parser.add_argument("--onnx", type=str,
                                         help="run ONNX model via TRT")
          self.parser.add_argument("--variational_inf_runs",
                                         type=int, default=0,
                                         help="no. runs for variational_inference.py")
          self.parser.add_argument("--variational_inf_path", type=str,
                                         help="path to save variational inf outputs")
     
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
        return self.opt   
        