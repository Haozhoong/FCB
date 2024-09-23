import torch
import torch.nn as nn
import numpy as np
from .stransconv import DeepSparse

def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j*real_img[1]
    elif axis == 1:
        complex_img = real_img[:,0] + 1j*real_img[:,1]
    else:
        raise NotImplementedError
    return complex_img

#CNN denoiser ======================
def conv_block_ds_fcb(in_channels, out_channels, num_rows, num_cols):
    return nn.Sequential(
        DeepSparse(in_channels, in_channels, num_rows, num_cols),
        nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode= 'circular'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def conv_block_ds(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False, padding_mode= 'circular'),
        nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode= 'circular'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_rows, num_cols, fcb=True):
        super().__init__()
        if fcb:
            self.layers = conv_block_ds_fcb(in_channels, out_channels, num_rows, num_cols)
        else:
            self.layers = conv_block_ds(in_channels, out_channels)
        if in_channels != out_channels:
            self.resample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.resample = nn.Identity()

    def forward(self, input):
        shortcut = self.resample(input)
        return self.layers(input) + shortcut

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers, num_rows, num_cols, fcb_l):
        super().__init__()
        layers = []
        layers += conv_block(2, 64)

        for i in range(n_layers-2):
            layers += nn.Sequential(ResBlock(64, 64, num_rows, num_cols, i>=n_layers-2-fcb_l))

        layers += nn.Sequential(
            nn.Conv2d(64, 2, 3, padding=1),
            nn.BatchNorm2d(2)
        )

        self.nw = nn.Sequential(*layers)
    
    def forward(self, x):
        idt = x # (2, nrow, ncol)
        dw = self.nw(x) + idt # (2, nrow, ncol)
        return dw

#CG algorithm ======================
class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam 

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im = im.unsqueeze(1)
        im_coil = self.csm * im # split coil images (B x ncoil x nrow x ncol)
        k_full = torch.fft.fft2(im_coil, dim=(-2,-1), norm='ortho') # convert into k-space 
        k_u = k_full * self.mask # undersampling
        im_u_coil = torch.fft.ifft2(k_u, dim=(-2,-1), norm='ortho') # convert into image domain
        im_u = torch.sum(im_u_coil * self.csm.conj(), axis=1) # coil combine (B x nrow x ncol)
        return im_u.squeeze(1) + self.lam * im.squeeze(1)

def myCG(AtA, rhs):
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    """
    rhs = r2c(rhs, axis=1) # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj()*r).real
    while i < 10 and rTr > 1e-10:
        Ap = AtA(p)
        alpha = rTr / torch.sum(p.conj()*Ap).real
        alpha = alpha
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj()*r).real
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return c2r(x, axis=1)

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k # (2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam)
        rec = myCG(AtA, rhs)
        return rec

#model =======================    
class FMoDL(nn.Module):
    def __init__(self, n_layers, k_iters, num_rows, num_cols, fcb_l=3):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers, num_rows, num_cols, fcb_l)
        self.dc = data_consistency()

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        
        x_k = x0.clone()
        for k in range(self.k_iters):
            #dw 
            z_k = self.dw(x_k) # (2, nrow, ncol)
            #dc
            x_k = self.dc(z_k, x0, csm, mask) # (2, nrow, ncol)
        return x_k

    def repara(self, model_load):
        # layercheck:
        check_flag = repara_modl_check(self, model_load)
        if not check_flag:
            print('Failed: Model Mismatched')
            return False
        
        # load same layers
        self.load_state_dict(model_load.state_dict(), strict=False)
        
        # load deepsparse layer
        for (name1, module1), (name2, module2) in zip(self.named_modules(), model_load.named_modules()):
            if isinstance(module1, DeepSparse):
                module1.loadweight(module2)
                
        return True

def repara_modl_check(model1, model2):
    check_flag = True
    for (name1, module1), (name2, module2) in zip(model1.named_modules(), model2.named_modules()):
        if name1 != name2:
            print('Mismatched module(name):  ', name1, name2)
            check_flag = False
            break

    try:
        for kb, cb in zip(model1.dw.nw, model2.dw.nw):
            if isinstance(kb, ResBlock):
                if kb.layers[1].in_channels != cb.layers[1].in_channels or \
                    kb.layers[1].out_channels != cb.layers[1].out_channels:
                        print('Mismatched module(channels):  ', kb, cb)
                        check_flag = False
    except:
        print('Couldn\'t find dw.nw')
        check_flag = False
        
    return check_flag