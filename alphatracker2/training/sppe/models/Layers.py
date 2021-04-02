import torch.nn as nn
import torch
import torch.nn.functional as F


##############################################################


#### #### #### #### Load pretrained models #### #### #### ####
##############################################################

mobilenet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).features




############################################################################


#### #### #### #### Squeeze and excitation layer prereqs #### #### #### ####
############################################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.1)
        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        
		
################################################################


#### #### #### #### Resnet 50 and 101 models #### #### #### ####
################################################################

class SEResnet101(nn.Module):
    """ SEResnet 101 """

    def __init__(self):
        super(SEResnet101, self).__init__()
        #assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        #self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.layers = [3,4,23,3]
        self.block = Bottleneck

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2)

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes,
                                stride, downsample, reduction=True))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
		

class SEResnet50(nn.Module):
    """ SEResnet 50 """

    def __init__(self):
        super(SEResnet50, self).__init__()
        #assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        #self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.layers = [3,4,6,3]
        self.block = Bottleneck

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2)

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes,
                                stride, downsample, reduction=True))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
		
		
#############################################################  
 
 
#### #### #### #### Upsampling layers DUC #### #### #### #### 	
#############################################################
        
class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x
        
		
##########################################################
	

#### #### #### #### Combination layers #### #### #### #### 
##########################################################
        
class mobilenet_backbone(nn.Module):
    # 128, 320, 1024, 256, 512 order
    #conv_dim = 128

    def __init__(self, nClasses, mode='large'):
        super(mobilenet_backbone, self).__init__()
        
        #self.preact = SEResnet('resnet101')
        self.preact = mobilenet

        self.suffle1 = nn.PixelShuffle(2)
        if mode == 'large':
            self.conv_dim = 128
            self.duc1_ = 320; self.duc2_ = 1024; self.duc3_ = 256; self.duc4_ = 512
        
        if mode == 'small':
            self.conv_dim = 64
            self.duc1_ = 320; self.duc2_ = 512; self.duc3_ = 128; self.duc4_ = 256
        
            
        self.duc1 = DUC(self.duc1_, self.duc2_, upscale_factor=2)
        self.duc2 = DUC(self.duc3_, self.duc4_, upscale_factor=2)
        self.nClasses = nClasses

        self.conv_out = nn.Conv2d(
            self.conv_dim, self.nClasses, kernel_size=3, stride=1, padding=1)
        #self.sig = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # print(self.duc1)
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        #out = self.sig(out)
        return out
		
				
class senet101_backbone(nn.Module):
    #conv_dim = 128

    def __init__(self, nClasses, mode='large'):
        super(senet101_backbone, self).__init__()

        self.preact = SEResnet101()
        
        self.suffle1 = nn.PixelShuffle(2)
        if mode == 'large':
            self.conv_dim = 128
            self.duc1_ = 512; self.duc2_ = 1024; self.duc3_ = 256; self.duc4_ = 512
        
        if mode == 'small':
            self.conv_dim = 64
            self.duc1_ = 512; self.duc2_ = 512; self.duc3_ = 128; self.duc4_ = 256
            
        self.duc1 = DUC(self.duc1_, self.duc2_, upscale_factor=2)
        self.duc2 = DUC(self.duc3_, self.duc4_, upscale_factor=2)
        self.nClasses = nClasses

        self.conv_out = nn.Conv2d(
            self.conv_dim, self.nClasses, kernel_size=3, stride=1, padding=1)
        #self.sig = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # print(self.duc1)
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        #out = self.sig(out)
        return out
		
		
class senet50_backbone(nn.Module):
    #conv_dim = 128

    def __init__(self, nClasses, mode='large'):
        super(senet50_backbone, self).__init__()

        self.preact = SEResnet50()
        
        self.suffle1 = nn.PixelShuffle(2)
        if mode == 'large':
            self.conv_dim = 128
            self.duc1_ = 512; self.duc2_ = 1024; self.duc3_ = 256; self.duc4_ = 512
        
        if mode == 'small':
            self.conv_dim = 64
            self.duc1_ = 512; self.duc2_ = 512; self.duc3_ = 128; self.duc4_ = 256
            
        self.duc1 = DUC(self.duc1_, self.duc2_, upscale_factor=2)
        self.duc2 = DUC(self.duc3_, self.duc4_, upscale_factor=2)
        self.nClasses = nClasses

        self.conv_out = nn.Conv2d(
            self.conv_dim, self.nClasses, kernel_size=3, stride=1, padding=1)
        #self.sig = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # print(self.duc1)
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        #out = self.sig(out)
        return out
