import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
from tqdm.notebook import tqdm
import time
import json

from torchsample.transforms import SpecialCrop, Pad
import torch.nn.functional as F
import imgaug

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from . import ImageUtil

def read_json(train_dir, train_json):
    #train_dir = r'C:\Users\lihao\Desktop\demo'
    #train_json = [r'C:\Users\lihao\Desktop\demo\train9.json']

    images = os.listdir(train_dir)

    with open(train_json[0], 'r') as f:
        data = json.load(f)
    
    return data
    
    
def clean_and_return_data(data, num_mouse, num_pose, train_val_split):
    ii=0
    num_badAnnot = 0
    num_allAnnot_train = 0
    num_allAnnot_valid = 0

    train_data = []
    valid_data = []

    new_data = []
    for i in data:
        single_img_data = i
        name=single_img_data['filename']
        annot=single_img_data['annotations']

        new_annot = []
        has_box = False
        for idx in range(len(annot)):
            if annot[idx]['class']=='Face' or annot[idx]['class']=='boundingBox' or annot[idx]['class']=='point':
                new_annot.append(annot[idx]) 
            if annot[idx]['class']=='Face' or annot[idx]['class']=='boundingBox':
                has_box = True
        annot = new_annot 
        if(len(new_annot)!=num_mouse[ii]*(num_pose+1)):
            print('Bad annotation: there %s animals and %s pose for each animal, but only %s annoation'%(str(num_mouse[ii]),str((num_pose+1)),str(len(new_annot))))
            num_badAnnot += 1
            continue
        if(not has_box):
            print('The annotations does not have info of bounding box')
            num_badAnnot += 1
            continue
        for mice_id in range(num_mouse[ii]):
            d=annot[mice_id*(num_pose+1)]
            bbox = [d['x'], d['y'], d['x']+d['width'], d['y']+d['height']][:]
            pt = [[annot[mice_id*(num_pose+1)+k+1]['x'], annot[mice_id*(num_pose+1)+k+1]['y']] for k in range(num_pose)][:]
            iname = [ord(x) for x in name][:]
            
        new_data.append(single_img_data)
        #single_img_data_count += 1

    train_data = new_data[0:int(len(new_data)*train_val_split)]
    val_data = new_data[int(len(new_data)*train_val_split):]
    print(len(train_data), len(val_data))
    
    return train_data, val_data
    
    
    
    
def load_into_memory(image_directory, num_mouse, num_pose, train_data):
    all_images = []
    all_labels = []
    all_masks = []
    all_points = []
    for i in tqdm(train_data):
        images, labels, masks, points = ImageUtil.main_generate_train(image_directory, num_mouse, num_pose, i)
        
        for i, j, k, m in zip(images, labels, masks, points):
            all_images.append(i.numpy())
            all_labels.append(j.numpy())
            all_masks.append(k.numpy())
            all_points.append(m)
            
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    all_masks = np.array(all_masks)
    all_points = np.array(all_points)

    all_images = np.transpose(all_images, (0, 2, 3, 1))
    all_labels = np.transpose(all_labels, (0, 2, 3, 1))
    all_masks = np.transpose(all_masks, (0, 2, 3, 1))
    #print(all_images.shape, all_labels.shape, all_masks.shape, all_points.shape)
    
    kps_ = []
    for j in all_points:
        kps = KeypointsOnImage([(Keypoint(x=int(j[i][0]), 
                                          y=int(j[i][1]))) for i in range(num_pose)], shape=(80, 64, 3))
        kps_.append(kps)
    #print(len(kps_))
    
    print("Images shape is: {}".format(all_images.shape))
    print("Labels shape is: {}".format(all_labels.shape))
    print("Masks shape is: {}".format(all_masks.shape))
    print("Points shape is: {}".format(len(kps_)))
    
    return all_images, all_labels, all_masks, all_points, kps_
        
        
        
    

    
