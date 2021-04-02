import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
from tqdm.notebook import tqdm
import time
import json
import random
import time

from torchsample.transforms import SpecialCrop, Pad
import torch.nn.functional as F
import imgaug
from datetime import datetime

from .utils.ImageUtil import drawGaussian, transformBox, transform,augmentation

seq = augmentation()
#from .models.FastPose import createModel


def border_pad(image, bordersize):
    img = cv2.copyMakeBorder(image, top=int(bordersize/2), 
                                    bottom=int(bordersize/2),
                                    left=bordersize,
                                    right=bordersize, 
                                    borderType=cv2.BORDER_CONSTANT,)
    return img


def visualize_progress(val_images, val_labels, preds):
    # stack images
    stacked_images = np.vstack(np.array([border_pad(i, 50) for i in val_images[0:2]]))
    
    # stack labels
    stacked_labels = np.array([border_pad(cv2.cvtColor(cv2.resize(i, (256, 320)).sum(axis=2),cv2.COLOR_GRAY2RGB), 50)
                               for i in val_labels[0:2]])
    stacked_labels = np.vstack(stacked_labels)
    
    # stack predictions
    stacked_predictions = preds.cpu().detach().numpy(); 
    stacked_predictions = stacked_predictions.sum(axis=1)
    
    stacked_predictions = np.array([border_pad(cv2.cvtColor(cv2.resize(i, (256, 320)),cv2.COLOR_GRAY2RGB), 50)
                               for i in stacked_predictions]); #print(stacked_predictions.shape)
    stacked_predictions = np.vstack(stacked_predictions)
    
    
    
    # concatenate all    
    horizontal = np.hstack((stacked_images, stacked_labels, stacked_predictions)); 
    return horizontal
    

def train_batch(m, chunks, all_images, all_masks, val_images, val_labels, kps_, criterion, optimizer, writer, e_num, vis=0, aug=1, ):
    #lossLogger = DataLogger()
    #accLogger = DataLogger()
    m.train()
    
    # each batch here
    chunks = tqdm(chunks)
    
    for batch_number, i in enumerate(chunks):
        t0 = time.time()
        i = np.array(i)
        i_ = i.tolist(); #t1 = time.time()
        kps_list = [kps_[j] for j in i]; 
        al = (all_images[i]*255).astype(np.uint8); t1 = time.time()
        ma = torch.from_numpy(all_masks[i]).float().cuda()
        if aug:
            image_aug, kps_aug = seq(images=al, keypoints=kps_list); 
        else:
            image_aug = al; 
            kps_aug = kps_list
        image_aug = (np.transpose(image_aug, (0, 3, 1, 2))/255.0).astype(np.float32); 
        p1 = time.time()

        
        image_aug2 = []
        for i in image_aug:
            o = transform(i)
            image_aug2.append(o)
        image_aug = np.array(image_aug2)
        #print(image_aug.shape)
        
        t_aug1 = time.time()
        labels_aug = []
        for kps_ind in kps_aug:
            o = []
            for ind in kps_ind[0:]:
                blank = np.zeros((80,64))
                blank = drawGaussian(blank, ind.xy, 1)
                o.append(blank)
            o = np.array(o)
            labels_aug.append(o)
        t_aug2 = time.time()

        c_t1 = time.time()
        labels_aug = (np.array(labels_aug)).astype(np.float32)  
        image_aug = torch.from_numpy(image_aug).cuda().requires_grad_()
        labels_aug = torch.from_numpy(labels_aug).cuda()
        c_t2 = time.time()

        
        # train network
        #t1 = time.time()
        out = m(image_aug)
        loss = criterion(out, labels_aug) #mul!!
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        
        print("loss:{loss:.8f}, time:{time:.8f} \r".format(loss=loss, time=(t1-t0)), end="")
        
        if vis:
            m_eval = m.eval()
            m_eval.cuda()
            
            #inp = val_images[0:2]
            inp = (np.transpose(val_images[0:2], (0, 3, 1, 2))).astype(np.float32); #print(inp.min(), inp.max())
            inp2 = []
            for i in inp:
                #print(i.max(), i.min())
                o = transform(i); 
                inp2.append(o)
            inp2 = np.array(inp2)
            inp2 = torch.from_numpy(inp2).cuda()
            #print(inp2.min(), inp2.max())
            
            with torch.no_grad():
                preds = m_eval(inp2)
            h = visualize_progress(val_images, val_labels, preds);
            #np.save('sppe_progress/batch_0000{}_{}'.format( e_num, batch_number), h) # optional saving
            cv2.imshow("frame", h)
            cv2.waitKey(1)
            
    cv2.destroyAllWindows()
    
    m_eval = m.eval()
    m_eval.cuda()
    
    inp = (np.transpose(val_images, (0, 3, 1, 2))).astype(np.float32); 
    inp2 = []
    for i in inp:
        #print(i.max(), i.min())
        o = transform(i); 
        inp2.append(o)
    inp2 = np.array(inp2)
    inp2 = torch.from_numpy(inp2).cuda()
    
    with torch.no_grad():
    	preds = m_eval(inp2)
    labs = np.transpose(val_labels, (0, 3, 1, 2)).astype(np.float32)
    val_loss = criterion(preds, torch.from_numpy(labs).cuda())
    print("epoch {}: train_loss = {}, val_loss = {}".format(e_num, loss, val_loss))
    
    return loss, val_loss

       
        
