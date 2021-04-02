import torch
import torch.nn as nn
from . import Layers


#def make_model(nClasses):
#    return Layers.FastPose_SE_pretrained(nClasses)
    
def make_model(nClasses, model):
    if model == 'senet_101':
        return Layers.senet101_backbone(nClasses, mode='large')
		
    if model == 'senet_50':
	    return Layers.senet50_backbone(nClasses, mode='large')

    if model == 'mobilenet':
        return Layers.mobilenet_backbone(nClasses, mode='large')