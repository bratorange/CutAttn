import os

import torch

from data import BaseDataset
from data.base_dataset import get_transform
from data.image_folder import make_dataset
from PIL import Image



class PretrainDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')
        transform = get_transform(self.opt)

        crop_size = self.opt.crop_size
        A = torch.rand(3, crop_size, crop_size)
        B = transform(B_img)
        return {'A': A, 'B': B, 'A_paths': "", 'B_paths': B_path}

    def __len__(self):

        return self.B_size