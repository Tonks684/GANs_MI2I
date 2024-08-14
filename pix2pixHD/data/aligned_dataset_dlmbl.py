import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
from pathlib import Path
from tifffile import imread
class AlignedDataset(BaseDataset):
    """
    A custom dataset class for loading aligned image datasets.

    Args:
        opt (argparse.Namespace): The command line arguments.
    """

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A
        dir_A = 'input'
        self.dir_A = os.path.join(opt.dataroot,dir_A, f'{opt.phase}')
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### target B 
        # if opt.isTrain or opt.phase == 'val':         
        self.dir_B = os.path.join(opt.dataroot,opt.target, f'{opt.phase}')
        self.B_paths = sorted(make_dataset(self.dir_B))

        assert len(self.B_paths) == len(self.A_paths), "The number of images in the input and target folders must be the same {} != {}".format(len(self.A_paths), len(self.B_paths))
        ### instance maps ()
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        self.dataset_size = len(self.A_paths)
    
    def minmax_norm(self, img, max, min):
        img = (img - min) / (max - min)
        return img.astype(np.float32)
        
    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input data and paths.
        """
        A_path = self.A_paths[index]
        A = imread(A_path)
        # No normalisation as already between -1 and 1
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params, normalize=False)
            A_tensor = transform_A(A)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = imread(B_path)
            if self.opt.target == 'nuclei':
                B = self.minmax_norm(B, 8603.0, 0.0)
                B = (B * 2) - 1
            elif self.opt.target == 'cyto':
                B = self.minmax_norm(B, 18372.0, 0.0)
                B = (B * 2) - 1
            else:
                raise ValueError("Unknown target")
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}
        return input_dict

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        """
        Returns the name of the dataset.

        Returns:
            str: The name of the dataset.
        """
        return "HEKCells Dataset"
