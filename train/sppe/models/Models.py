import torch
import torch.nn as nn
from . import Layers


def make_model(nClasses):
    return Layers.FastPose_SE_pretrained(nClasses)





