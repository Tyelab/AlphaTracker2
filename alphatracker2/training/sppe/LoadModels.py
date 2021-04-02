import torchvision
import torch
from torch import nn
from .models import Models

import os
import os, contextlib


class inference_model_fast(nn.Module):
    def __init__(self, model, nClasses):
        super(inference_model_fast, self).__init__()

        self.pyranet = model
        self.nClasses = nClasses


    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, self.nClasses)

        return out
        
        
def pose_estimator(nClasses, model_type, trained=''):

    ''' load the pose estimator
    
        nClasses: number of bodyparts being tracked
        model_type: type of model that user has selected, including
                    'mobilenet', 'senet101', 'senet50'
        trained: string for the path to the trained model, typically a '.pkl' file
        
    '''

    model = Models.make_model(nClasses, model_type)
    #print(torch.cuda.is_available())
	
    if torch.cuda.is_available():
        if trained:
            model.load_state_dict(torch.load(trained)).cuda()
        else:
            model = model.cuda()
        #model.eval()
        
        print('cuda:True...using GPU')
    else:
        if trained:
            model.load_state_dict(torch.load(trained, map_location='cpu'))
        else:
            model = model.to('cpu')
        #model.eval()
        print('cuda:False...using CPU')
        
    model = inference_model_fast(model, nClasses)
    
    return model
	
	
