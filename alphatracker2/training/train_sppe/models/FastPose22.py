# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torchvision
import torch
import torch.nn as nn

from .layers.DUC import DUC
from .layers.SE_Resnet import SEResnet, SEResnet50

# Import training option
from ..opt import opt


mobilenet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).features

def createModel(model_type, nClasses):
    if model_type=='senet_101':
        return FastPose_SE(nClasses)
        
    if model_type=='senet_50':
        return resnet50_backbone(nClasses)
        
    if model_type=='mobilenet':
        return mobilenet_backbone(nClasses)


class FastPose_SE(nn.Module):
    conv_dim = 128

    def __init__(self, nClasses):
        super(FastPose_SE, self).__init__()
        
        self.nClasses = nClasses; self.preact = SEResnet('resnet101'); self.suffle1 = nn.PixelShuffle(2); 
        self.duc1 = DUC(512, 1024, upscale_factor=2);  self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(self.conv_dim, self.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # print(self.duc1)
        out = self.preact(x); out = self.suffle1(out); out = self.duc1(out); out = self.duc2(out)
        out = self.conv_out(out)
        return out

        
class resnet50_backbone(nn.Module):
    conv_dim = 128
    
    def __init__(self, nClasses):
        super(resnet50_backbone, self).__init__()
        
        self.nClasses = nClasses; self.preact = SEResnet50(); self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2); self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(self.conv_dim, self.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # print(self.duc1)
        out = self.preact(x); out = self.suffle1(out); out = self.duc1(out); out = self.duc2(out)

        out = self.conv_out(out)
        return out
        
        
class mobilenet_backbone(nn.Module):
    conv_dim = 128
    
    def __init__(self, nClasses):
        super(mobilenet_backbone, self).__init__()
        
        self.nClasses = nClasses; self.preact = mobilenet; self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(320, 1024, upscale_factor=2); self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(self.conv_dim, self.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # print(self.duc1)
        out = self.preact(x); out = self.suffle1(out); out = self.duc1(out); out = self.duc2(out)

        out = self.conv_out(out)
        return out