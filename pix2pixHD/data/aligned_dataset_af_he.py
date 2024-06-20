import os.path
#from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.my_base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
from tifffile import imread


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (AFs)
        dir_A = '_A' 
        self.dir_A1 = os.path.join(opt.dataroot,'Cy3', opt.phase + dir_A)
        self.A1_paths = sorted(make_dataset(self.dir_A1))
        self.dir_A2 = os.path.join(opt.dataroot, 'DAPI',opt.phase + dir_A)
        self.A2_paths = sorted(make_dataset(self.dir_A2))

        ### input B (real images)
        if opt.isTrain or opt.phase == 'val':
            dir_B = '_B' 
            self.dir_B = os.path.join(opt.dataroot, 'HE', opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        self.dataset_size = len(self.A1_paths)

    def __getitem__(self, index):
        ### input A1 Cy3 AFs 32float 0 to 1 RGB but gray copied on RGB
        A1_path = self.A1_paths[index]
        A1 = imread(A1_path)
        A1 = A1[:,:,0] # take single gray channel
        A1 = A1[:,:,np.newaxis]
        A1 = A1.astype(np.float32)        
        params = get_params(self.opt, A1)
        transform_A = get_transform(self.opt, params,normalize=False)
        A1_tensor = transform_A(A1)
        ### input A2 DAPI AFs 32float 0 to 1 
        A2_path = self.A2_paths[index]
        A2 = imread(A2_path)
        A2 = A2[:,:,0] # take single gray channel
        A2 = A2[:,:,np.newaxis] #256,256,1
        A2 = A2.astype(np.float32)
        transform_A = get_transform(self.opt, params,normalize=False)
        A2_tensor = transform_A(A2)
       
        ### input B (real images)
        B_tensor = inst_tensor = feat_tensor = 0
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = imread(B_path)
            B = B.astype(np.float32)
            transform_B = get_transform(self.opt, params,normalize=False)
            B_tensor = transform_B(B)

        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {
            'af_cy3': A1_tensor,
            'af_dapi': A2_tensor,
            'inst': inst_tensor, 
            'image': B_tensor,
            'feat': feat_tensor, 
            'path': A1_path}
        
        return input_dict

    def __len__(self):
        return len(self.A1_paths) // self.opt.batchSize * self.opt.batchSize
    
    
    def name(self):
        return "Autoflorescence to H&E Dataset"
