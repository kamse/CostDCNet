

import os
import warnings
import numpy as np
import json
import h5py
# from . import BaseDataset
from torch.utils.data import Dataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import copy
warnings.filterwarnings("ignore", category=UserWarning)


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

# Reference : https://github.com/zzangjinsun/NLSPN_ECCV20
class NYU(BaseDataset):
    def __init__(self, args, mode, augment = True):
        super(NYU, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.data_path = args.data_path
        # self.data_path = "/hdd2/NYUv2/Fangchang_Ma/nyudepthv2"
        # if os.getcwd().split('/')[1] == 'root':
        #     self.data_path = "/root/soongjin/data_local/kimsj0302/NYUv2/Fangchang_Ma/nyudepthv2"
                
        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # For NYUDepthV2, crop size is fixed
        self.height = 240
        self.width = 320
        self.crop_size = (228, 304)
        self.augment = augment

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        with open('./splits/nyu.json') as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]
        
        self.resizer  = T.Resize((self.crop_size[0] // args.up_scale, self.crop_size[1] // args.up_scale), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        inputs = {}
        
        path_file = os.path.join(self.data_path, self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')


        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)
            do_color_aug = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            if do_color_aug > 0.5:
                rgb = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(rgb)

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)

            t_rgb = T.Compose([
                T.Resize(scale),
                # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

        else:
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            
        inputs[("color_aug", 1, 0)] = rgb
        inputs[("depth_gt", 1, 0)] = dep
        inputs[("mask", 1, 0)] = (inputs[("depth_gt", 1, 0)] > 0).type(torch.float32)
        inputs[("sp_depth", 1, 0)] = self.get_sparse_depth(dep)

        return inputs

    def get_sparse_depth(self, dep, num_sample = 500):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp