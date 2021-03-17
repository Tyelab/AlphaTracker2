import torchvision
import torch
from torch import nn
from .sppe.models import Models

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
        
        
def pose_estimator(nClasses, model_type, trained):

    ''' load the pose estimator
    
        nClasses: number of bodyparts being tracked
        model_type: type of model that user has selected, including
                    'mobilenet', 'senet101', 'senet50'
        trained: string for the path to the trained model, typically a '.pkl' file
        
    '''

    model = Models.make_model(nClasses, model_type)
    #print(torch.cuda.is_available())
	
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(trained)).cuda()
        model.eval()
        
        print('cuda:True...using GPU')
    else:
        model.load_state_dict(torch.load(trained, map_location='cpu'))
        model.eval()
        print('cuda:False...using CPU')
        
    model = inference_model_fast(model, nClasses)
    
    return model
	
	
def object_detector(trained):

    ''' load the object detector in this step
        
        trained: string for the path to the trained model, typically a YOLO model
    
    '''
    
    #with open(os.devnull, 'w') as devnull:
    #    with contextlib.redirect_stdout(devnull):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=trained, verbose=False)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    model.eval()
    return model