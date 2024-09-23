import torch
import torch.nn as nn
import numpy as np

def c2r(complex_img, axis=0):
    """
    :input shape: 1 x row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.concatenate((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.cat((complex_img.real, complex_img.imag), dim=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(images, axis=1):
    """
    :input shape: 2c x row x col (float32)
    :output shape: 1c x row x col (complex64)
    """
    C = int(images.shape[axis]/2)
    images = torch.complex(torch.index_select(images, axis, torch.tensor(range(C), device=images.device)),
                           torch.index_select(images, axis, torch.tensor(range(C, images.shape[axis]), device=images.device)))
    return images

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

class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None, shift=False):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))
        self.shift = shift

    def perform(self, x, k0, mask, sensitivity):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = r2c(x, axis=1)
        x = x * sensitivity
        
        k = img2k(x, shift=self.shift)
              
        v = self.noise_lvl
        if v is not None: # noisy case
            # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0) 
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0
    
        # ### backward op ### #
        x = k2img(out, shift=self.shift)
       
        Sx = (x*sensitivity.conj()).sum(dim=1, keepdim=True)
        Sx = c2r(Sx, axis=1)
        SS = (sensitivity*sensitivity.conj()).sum(dim=1, keepdim=True)
        SS = c2r(SS, axis=1)
   
        return Sx, SS

    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        
        x = self.para*cnn + (1 - self.para)*Sx
        return x

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU()
    )

def conv_block_ds(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False, padding_mode= 'circular'),
        nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode= 'circular'),
        nn.ReLU()
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dwconv=True):
        super().__init__()
        if dwconv:
            self.layers = conv_block_ds(in_channels, out_channels)
        else:
            self.layers = conv_block(in_channels, out_channels)
            
        if in_channels != out_channels:
            self.resample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.resample = nn.Identity()

    def forward(self, input):
        shortcut = self.resample(input)
        return self.layers(input) + shortcut

class cnn_layer(nn.Module):
    
    def __init__(self, dwconv=False):
        super(cnn_layer, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        for i in range(3):
            self.conv.append(ResBlock(64, 64,  dwconv))
        
        self.conv.append(nn.Conv2d(64, 2,  3, padding=1, bias=True))
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class VSNet(nn.Module):
    
    def __init__(self, alfa=1, beta=1, cascades=5, dwconv=False, shift=False):
        super(VSNet, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(cnn_layer(dwconv)) 
            dc_blocks.append(dataConsistencyTerm(alfa, shift)) 
            wa_blocks.append(weightedAverageTerm(beta)) 
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        
 
    def forward(self, x, k, m, c):
                
        for i in range(self.cascades):
            x_cnn = self.conv_blocks[i](x)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c)
            #print('x_cnn', x_cnn.shape)
            #print('Sx', Sx.shape)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)
        return x