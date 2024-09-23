import torch
import torch.nn as nn
import torch.nn.init as init
from .stransconv import DeepSparse
    
class double_conv_dw(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, input_channels, output_channels, shorcut=True, circ_pad=True):
        super(double_conv_dw, self).__init__()
        padding_mode = 'circular' if circ_pad else 'zero'
        self.depthwise1 = nn.Conv2d(input_channels, input_channels, 3, padding=1,
                                    groups=input_channels, bias=False, padding_mode=padding_mode)
        self.pointwise1 = nn.Conv2d(input_channels, output_channels, 1, padding=0, bias=False)
        self.depthwise2 = nn.Conv2d(output_channels, output_channels, 3, padding=1,
                                    groups=output_channels, bias=False, padding_mode=padding_mode)
        self.pointwise2 = nn.Conv2d(output_channels, output_channels, 1, padding=0, bias=False)
        self.shortcut_flag = shorcut
        self.shortcut = nn.Conv2d(input_channels, output_channels, 1, padding=0, bias=False)
        self.relu = nn.PReLU()
        
    def forward(self, x):
        res1 = self.depthwise1(x)
        if self.shortcut_flag:
            res1 = res1 + x
        res1 = self.pointwise1(res1)
        res1 = self.relu(res1)
        res2 = self.depthwise2(res1)
        if self.shortcut_flag:
            res2 = res2 + res1
        res2 = self.pointwise2(res2)
        if self.shortcut_flag:
            res2 =  res2 + self.shortcut(x)
        
        return res2
    
class fdouble_conv(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, input_channels, output_channels, num_rows, num_cols, shorcut=True):
        super(fdouble_conv, self).__init__()
        self.depthwise1 = DeepSparse(input_channels, input_channels, num_rows, num_cols)
        self.pointwise1 = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.depthwise2 = DeepSparse(output_channels, output_channels, num_rows, num_cols)
        self.pointwise2 = nn.Conv2d(output_channels, output_channels, 1, bias=False)
        self.shortcut = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.relu = nn.PReLU()
        self.shortcut_flag = shorcut
        
    def forward(self, x):
        res1 = self.depthwise1(x)
        if self.shortcut_flag:
            res1 = res1 + x
        res1 = self.pointwise1(res1)
        res1 = self.relu(res1)
        res2 = self.depthwise2(res1)
        if self.shortcut_flag:
            res2 = res2 + res1
        res2 = self.pointwise2(res2)
        if self.shortcut_flag:
            res2 =  res2 + self.shortcut(x)
        
        return res2

class down(nn.Module):
    def __init__(self, in_ch, out_ch, shorcut=True, fouier=False,
                 num_rows=320, num_cols=320):
        super(down, self).__init__()
        
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),)
        
        if fouier:
            conv = fdouble_conv(in_ch, out_ch, num_rows, num_cols, shorcut=shorcut)
        else:
            conv = double_conv_dw(in_ch, out_ch, shorcut=shorcut)

        self.mpconv.append(conv)
        
    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_ch, out_ch, shorcut=True, fouier=False,
                 num_rows=320, num_cols=320, Transpose=False):
        super(up, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        else:

            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, padding=0),
                                    nn.PReLU())
        if fouier:
            self.conv = fdouble_conv(in_ch, out_ch, num_rows, num_cols, shorcut=shorcut)
        else:
            self.conv = double_conv_dw(in_ch, out_ch, shorcut=shorcut)
        
        
        self.up.apply(self.init_weights)
    def forward(self, x1, x2):
        '''
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


class inconv(nn.Module):
    ''' input conv layer
        let input channels image to 64 channels
        The only difference between `inconv` and `down` is maxpool layer
    '''
    def __init__(self, in_ch, out_ch, shorcut=True, fouier=False, num_rows=320, num_cols=320):
        super(inconv, self).__init__()

        if fouier:
            self.conv = fdouble_conv(in_ch, out_ch, num_rows, num_cols, shorcut=shorcut)
        else:
            self.conv = double_conv_dw(in_ch, out_ch, shorcut=shorcut)

        
    def forward(self, x):
        x = self.conv(x)
        return x


# for Re-Parameterization
def copydc(mk, mi):
    mk.depthwise1.loadweight(mi.depthwise1)
    mk.depthwise2.loadweight(mi.depthwise2)

def copydown(mk, mi):
    copydc(mk.mpconv[1], mi.mpconv[1])
    
def copyup(mk, mi):
    copydc(mk.conv, mi.conv)

class FUNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels=64,
                 num_rows=320, num_cols=320,
                 shorcut=True, res_study=False,
                 fcb=1, forward=False):
        
        super(FUNet, self).__init__()
        self.inc = inconv(in_ch, channels, shorcut, False, num_rows, num_cols)
        num_rows, num_cols = int(num_rows/2), int(num_cols/2)
        self.down1 = down(channels, channels*2, shorcut, fcb<1, num_rows, num_cols)
        num_rows, num_cols = int(num_rows/2), int(num_cols/2)
        self.down2 = down(channels*2, channels*4, shorcut, fcb<2, num_rows, num_cols)
        num_rows, num_cols = int(num_rows/2), int(num_cols/2)
        self.down3 = down(channels*4, channels*8, shorcut, fcb<3, num_rows, num_cols)
        num_rows, num_cols = int(num_rows/2), int(num_cols/2)
        self.down4 = down(channels*8, channels*16, shorcut, fcb<4, num_rows, num_cols)
        num_rows, num_cols = int(num_rows*2), int(num_cols*2)

        self.up1 = up(channels*16, channels*8, shorcut, (fcb<5 or forward) and fcb<5, num_rows, num_cols, False)
        num_rows, num_cols = int(num_rows*2), int(num_cols*2)
        self.up2 = up(channels*8, channels*4, shorcut, (fcb<6 or forward) and fcb<6, num_rows, num_cols, False)
        num_rows, num_cols = int(num_rows*2), int(num_cols*2)
        self.up3 = up(channels*4, channels*2, shorcut, (fcb<7 or forward) and fcb<7, num_rows, num_cols, False)
        num_rows, num_cols = int(num_rows*2), int(num_cols*2)
        self.up4 = up(channels*2, channels, shorcut, (fcb<8 or forward) and fcb<8, num_rows, num_cols, False)
        
        self.outc = nn.Conv2d(channels, out_ch, 1, padding=0, padding_mode= 'circular')
        self.res_study = res_study
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

    def forward(self, x):
        x0 = x.clone().detach()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.res_study:
            x = x + self.shortcut(x0)
        return x
    
    def repara(self, model_load):
        # layercheck:
        check_flag = repara_unet_check(self, model_load)
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
        
def repara_unet_check(model1, model2):
    check_flag = True
    for (name1, module1), (name2, module2) in zip(model1.named_modules(), model2.named_modules()):
        if name1 != name2:
            print('Mismatched module(name):  ', name1, name2)
            check_flag = False
            break
        
        if isinstance(module1, torch.nn.modules.container.Sequential):
            for (submodule1, submodule2) in zip(module1, module2):
                #print(submodule1)
                if isinstance(submodule1, double_conv_dw) or isinstance(submodule1, fdouble_conv):
                    # check circ_pad, dw, shortcut
                    if submodule1.shortcut_flag != submodule2.shortcut_flag:
                        print('Mismatched module(shortcut):  ', submodule1, submodule2)
                        check_flag = False
                    
                    if 'dw' not in str(type(submodule2)) or \
                        submodule2.depthwise1.padding_mode != 'circular':
                        print('Mismatched module(padding):  ', submodule1, submodule2)
                        check_flag = False
                    
                    if submodule1.pointwise1.in_channels != submodule2.pointwise1.in_channels or \
                        submodule1.pointwise1.out_channels != submodule2.pointwise1.out_channels:
                        print('Mismatched module(channels):  ', submodule1, submodule2)
                        check_flag = False
                        
    return check_flag