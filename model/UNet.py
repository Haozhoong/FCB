import torch
import torch.nn as nn
import torch.nn.init as init

class LConv2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'circular',  # TODO: refine this type
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding,
                                    groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_normal(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, input_channels, output_channels, kernel_size=3, shorcut=True, circ_pad=True):
        super(double_conv_normal, self).__init__()
        padding_mode = 'circular' if circ_pad else 'zeros'
        if kernel_size > 3:
            self.conv = nn.Sequential(
                LConv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2,
                        bias=False, padding_mode=padding_mode),
                nn.PReLU(),
                LConv2d(output_channels, output_channels, kernel_size, padding=1,
                        bias=False, padding_mode=padding_mode),
                nn.PReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2,
                          bias=False, padding_mode=padding_mode),
                nn.PReLU(),
                nn.Conv2d(output_channels, output_channels, kernel_size, padding=1,
                          bias=False, padding_mode=padding_mode),
                nn.PReLU()
            )
        self.shortcut_flag = shorcut
        self.shortcut = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        res = self.conv(x)
        if self.shortcut_flag:
            res = res + self.shortcut(x)
        return res
    
class double_conv_dw(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, input_channels, output_channels, kernel_size=3, shorcut=True, circ_pad=True):
        super(double_conv_dw, self).__init__()
        padding_mode = 'circular' if circ_pad else 'zeros'
        if kernel_size > 3:
            self.depthwise1 = LConv2d(input_channels, input_channels, kernel_size, padding=kernel_size//2,
                                      groups=input_channels, bias=False, padding_mode=padding_mode)
        else:
            self.depthwise1 = nn.Conv2d(input_channels, input_channels, kernel_size, padding=1,
                                        groups=input_channels, bias=False, padding_mode=padding_mode)
            
        self.pointwise1 = nn.Conv2d(input_channels, output_channels, 1, padding=0, bias=False)
        if kernel_size > 3:
            self.depthwise2 = LConv2d(output_channels, output_channels, kernel_size, padding=kernel_size//2,
                                      groups=output_channels, bias=False, padding_mode=padding_mode)
        else:
            self.depthwise2 = nn.Conv2d(output_channels, output_channels, kernel_size, padding=1,
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

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dwconv=True, shorcut=True, circ_pad=True):
        super(down, self).__init__()
        
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),)
        
        if dwconv:
            conv = double_conv_dw(in_ch, out_ch, kernel_size, shorcut, circ_pad)
        else:
            conv = double_conv_normal(in_ch, out_ch, kernel_size, shorcut, circ_pad)

        self.mpconv.append(conv)
        
    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3, Transpose=False, dwconv=True, shorcut=True, circ_pad=True):
        super(up, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        else:

            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, padding=0),
                                    nn.PReLU())
        if dwconv:
            self.conv = double_conv_dw(in_ch, out_ch, kernel_size, shorcut, circ_pad)
        else:
            self.conv = double_conv_normal(in_ch, out_ch, kernel_size, shorcut, circ_pad)
        
        
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
    def __init__(self, in_ch, out_ch, kernel_size=3, dwconv=True, shorcut=True, circ_pad=True):
        super(inconv, self).__init__()
        if dwconv:
            self.conv = double_conv_dw(in_ch, out_ch, kernel_size, shorcut, circ_pad)
        else:
            self.conv = double_conv_normal(in_ch, out_ch, kernel_size, shorcut, circ_pad)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels=64,
                 dwconv=True, shorcut=True,
                 circ_pad=True, res_study=False,
                 kernel_size=3, lcb=0, forward=False):
        super(UNet, self).__init__()
        self.inc = inconv(in_ch, channels, 3, dwconv, shorcut, circ_pad)
        self.down1 = down(channels*1, channels*2, kernel_size if lcb<1 else 3, dwconv, shorcut, circ_pad)
        self.down2 = down(channels*2, channels*4, kernel_size if lcb<2 else 3, dwconv, shorcut, circ_pad)
        self.down3 = down(channels*4, channels*8, kernel_size if lcb<3 else 3, dwconv, shorcut, circ_pad)
        self.down4 = down(channels*8, channels*16, kernel_size if lcb<4 else 3, dwconv, shorcut, circ_pad)

        self.up1 = up(channels*16, channels*8, kernel_size if (lcb<4 or forward) else 3, False, dwconv, shorcut, circ_pad)
        self.up2 = up(channels*8, channels*4, kernel_size if (lcb<3 or forward) else 3, False, dwconv, shorcut, circ_pad)
        self.up3 = up(channels*4, channels*2, kernel_size if (lcb<2 or forward) else 3, False, dwconv, shorcut, circ_pad)
        self.up4 = up(channels*2, channels, kernel_size if (lcb<1 or forward) else 3, False, dwconv, shorcut, circ_pad)
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
        
        # load same layers
        self.load_state_dict(model_load.state_dict(), strict=False)
        
        copydown(self.down3, model_load.down3)
        copydown(self.down4, model_load.down4)
        copyup(self.up1, model_load.up1)
        copyup(self.up2, model_load.up2)
        copyup(self.up3, model_load.up3)
        copyup(self.up4, model_load.up4)
        return True
    
def copydc(mk, mi):
    copyl(mk.depthwise1.conv, mi.depthwise1)
    copyl(mk.depthwise2.conv, mi.depthwise2)

def copydown(mk, mi):
    copydc(mk.mpconv[1], mi.mpconv[1])
    
def copyup(mk, mi):
    copydc(mk.conv, mi.conv)
    
def copyl(c1, c2):
    torch.nn.init.zeros_(c1.weight)
    dif = [c2.kernel_size[0] // 2, c2.kernel_size[1] // 2]
    cx = [c1.kernel_size[0] // 2 - dif[0], c1.kernel_size[0] // 2 + dif[0] + 1]
    cy = [c1.kernel_size[1] // 2 - dif[1], c1.kernel_size[1] // 2 + dif[1] + 1]
    c1.weight.data[..., cx[0]:cx[1], cy[0]:cy[1]] = c2.weight.detach().clone()