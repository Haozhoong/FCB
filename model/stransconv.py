import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState

def complexinit(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols = weights_real.shape
    fan_in = input_chs
    fan_out = output_chs
    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState()
    kernel_shape = weights_real.shape
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weights_real.data = torch.Tensor(weight_real)
    weights_imag.data = torch.Tensor(weight_imag)

class DeepSparse(nn.Module):
    def __init__(self, input_chs:int, output_chs:int, num_rows:int, num_cols:int, stride=1, init='he'):
        super(DeepSparse, self).__init__()
        self.weights_real = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols//2 + 1)))
        self.weights_imag = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols//2 + 1)))
        complexinit(self.weights_real, self.weights_imag, init)
        self.size = (num_rows, num_cols)
        self.stride = stride

    def forward(self, x):
        x = torch.fft.rfftn(x, dim=(-2, -1), norm=None)
        x_real, x_imag = x.real, x.imag
        y_real = torch.mul(x_real, self.weights_real) - torch.mul(x_imag, self.weights_imag)
        y_imag = torch.mul(x_real, self.weights_imag) + torch.mul(x_imag, self.weights_real)
        x = torch.fft.irfftn(torch.complex(y_real, y_imag), s=self.size, dim=(-2, -1), norm=None)
        if self.stride == 2:
            x = x[...,::2,::2]
        return x
        
    def loadweight(self, ilayer):
        weight = ilayer.weight.detach().clone()
        fft_shape = self.weights_real.shape[-2]
        weight = torch.flip(weight, [-2, -1])
        pad = torch.nn.ConstantPad2d(padding=(0, fft_shape - weight.shape[-1], 0, fft_shape - weight.shape[-2]),
                             value=0)
        weight = pad(weight)
        weight = torch.roll(weight, (-1, -1), dims=(-2, - 1))
        weight_kc = torch.fft.fftn(weight, dim=(-2, -1), norm=None).transpose(0, 1)
        weight_kc = weight_kc[..., :weight_kc.shape[-1] // 2 + 1]
        self.weights_real.data = weight_kc.real
        self.weights_imag.data = weight_kc.imag
