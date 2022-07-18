"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/jphdotam/Unet3D/blob/main/unet3d.py
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, f_maps = [32, 64, 96, 128], mode='trilinear', convtype = 1):
        super(UNet3D, self).__init__()
        self.convtype = [Conv3D, P3D][convtype]

        self.inc = DoubleConv(n_channels, f_maps[0], conv_type=self.convtype)
        self.down1 = Down(f_maps[0], f_maps[1], conv_type=self.convtype)
        self.down2 = Down(f_maps[1], f_maps[2], conv_type=self.convtype)
        self.down3 = Down(f_maps[2], f_maps[3], conv_type=self.convtype)
        # self.down4 = Down(f_maps[3], f_maps[4] // factor, conv_type=self.convtype)
        # self.up1 = Up(f_maps[4], f_maps[3] // factor, trilinear)
        self.up2 = Up(f_maps[3] + f_maps[2], f_maps[2], mode = mode, conv_type=self.convtype)
        self.up3 = Up(f_maps[2] + f_maps[1], f_maps[1], mode = mode, conv_type=self.convtype)
        self.up4 = Up(f_maps[1] + f_maps[0], f_maps[0], mode = mode, conv_type=self.convtype)
        self.classif0 = nn.Conv3d(f_maps[0], n_classes, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.classif0(x)
        return logits

class Conv3D(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(Conv3D, self).__init__()
        self.conv = nn.Conv3d(nin, nout, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(nout)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

# Reference : https://github.com/qijiezhao/pseudo-3d-pytorch/blob/master/p3d_model.py
class P3D(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(P3D, self).__init__()
        self.conv1 = nn.Conv3d(nin,nout,kernel_size=(1,3,3),stride=1, padding=(0,1,1),bias=False) 
        self.bn1 = nn.BatchNorm3d(nout)
        self.conv2 = nn.Conv3d(nout,nout,kernel_size=(3,1,1),stride=1,padding=(1,0,0),bias=False) 
        self.bn2 = nn.BatchNorm3d(nout)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=P3D):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, mid_channels = in_channels, conv_type=conv_type)
            # DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mode='trilinear', conv_type=P3D):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels, mid_channels=out_channels, conv_type=conv_type)
        self.up_mode = mode

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode=self.up_mode)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=Conv3D, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)

import numpy as np

if __name__ == "__main__":
    model = UNet3D(32, 1, convtype = 1).cuda()
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("num param:", params)
    
    l = torch.rand((3, 32, 8, 32, 32)).cuda()
    
    out1  = model(l)
    out1