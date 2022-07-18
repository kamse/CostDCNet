# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock

# from models.resnet import ResNetBase


class Encoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, planes= (32, 64, 96)):
        nn.Module.__init__(self)
        self.BLOCK = BasicBlock 
        self.D = D
        self.PLANES = planes 
        self.LAYERS = (1, 1, 1)
        self.inplanes = 32

        self.conv1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=3, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes) 
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride = 1)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride = [1,2,2])
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride = [1,2,2])

        self.conv2 = ME.MinkowskiConvolution(self.PLANES[2], out_channels, kernel_size=1, dimension=D)

        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)
       
    def forward(self, x):
        # feat = []
        out = self.conv1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        
        out_p2 = self.block1(out_p1)
        out_p3 = self.block2(out_p2)
        out_p4 = self.block3(out_p3)

        fin = self.conv2(out_p4)

        return fin