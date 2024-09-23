import numpy as np
import torch
import os
from torch.utils.data import Dataset

def default_loader(path):
    im = np.load(path)
    return im

def img2k(data, shift=True):
    if shift:
        data = torch.fft.fftshift(data, dim=(-2,-1))
    data = torch.fft.fft2(data, dim=(-2,-1), norm='ortho')
    if shift:
        data = torch.fft.ifftshift(data, dim=(-2,-1))
        
    return data

def k2img(data, shift=True):
    if shift:
        data = torch.fft.fftshift(data, dim=(-2,-1))
    data = torch.fft.ifft2(data, dim=(-2,-1), norm='ortho')
    if shift:
        data = torch.fft.ifftshift(data, dim=(-2,-1))
        
    return data

def tensormask(data, mask, noise_=False, noise_level=0.1, noise_seed=0, shift=True):
    data_k = img2k(data, shift)
    
    if noise_:
        g = torch.Generator()
        g.manual_seed(noise_seed)
        noise_s = torch.randn(data_k.shape[1], data_k.shape[2], generator=g)
        std = torch.std(data_k, dim=(-2,-1), keepdim=True) * noise_level
        noise = noise_s.expand([data_k.shape[0], data_k.shape[1], data_k.shape[2]]) * std
        data_k = data_k + noise
    
    data_k = data_k * mask
    data = k2img(data_k, shift)
    
    return data, data_k

class MyDataset(Dataset):
    def __init__(self, directory_full, directory_m, mask_path, label_index,
                 noise_=False, noise_level=0.1, shift=True, return_shift=True):
        # load full
        fh = os.listdir(directory_full)
        fh.sort()
        fn_t = [os.path.join(directory_full, k) for k in fh]
        fn_t = fn_t[label_index[0]:label_index[1] + 1]
        
        fh = os.listdir(directory_m)
        fh.sort()
        fn_m = [os.path.join(directory_m, k) for k in fh]
        fn_m = fn_m[label_index[0]:label_index[1] + 1]
        
        mask = default_loader(mask_path)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        if not shift:
            mask = torch.fft.fftshift(mask, dim=(-2,-1))
            
        if return_shift:
            mask_shift = torch.fft.fftshift(mask, dim=(-2,-1))
        else:
            mask_shift = mask.clone()
            
        self.fn_t = fn_t
        self.fn_m = fn_m
        self.mask = mask
        self.mask_shift = mask_shift
        self.shift = shift
        self.noise_ = noise_
        self.noise_level = noise_level

    def __getitem__(self, index):
        full_k = default_loader(self.fn_t[index])
        map = default_loader(self.fn_m[index])
        
        full = torch.from_numpy(full_k).cfloat()
        map = torch.from_numpy(map).cfloat()
        
        zf, measure = tensormask(full, self.mask, noise_=self.noise_,
                                 noise_level=self.noise_level, noise_seed=index*17, shift=self.shift)
        zf_single = torch.sum(zf * torch.conj(map), dim=0, keepdim=True)
        target_single = abs(torch.sum(full * torch.conj(map), dim=0, keepdim=True))
        
        return {'target_multi':full, 'target_single':target_single, 'input_single':zf_single,
                'input_multi':zf, 'sensmap':map, 'mask':self.mask_shift, 'measure':measure}

    def __len__(self):
        return len(self.fn_t)
    