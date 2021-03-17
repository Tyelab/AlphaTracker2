import torch
import torch.nn as nn
from . import Layers


def make_model(nClasses, model):
    if model == 'senet_101':
        return Layers.senet101_backbone(nClasses)
		
    if model == 'senet_50':
	    return Layers.senet50_backbone(nClasses)

    if model == 'mobilenet':
        return Layers.mobilenet_backbone(nClasses)





