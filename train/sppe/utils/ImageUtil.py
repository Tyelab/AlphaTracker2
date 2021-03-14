import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
from tqdm.notebook import tqdm
import time
import json
import random

from torchsample.transforms import SpecialCrop, Pad
import torch.nn.functional as F
import imgaug

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage



def cropBox(img, ul, br, resH, resW):
    ul = ul.int()
    br = br.int()
    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    newDim = torch.IntTensor((img.size(0), int(lenH), int(lenW)))

    if(ul[1]>=img.shape[1] or ul[0]>=img.shape[2]):
        print('This error may because yolo is not trained correctly or the weights is not used correctly.')
        raise IndexError

    newImg = img[:, ul[1]:, ul[0]:].clone()
    # Crop and Padding
    size = torch.IntTensor((int(br[1] - ul[1]), int(br[0] - ul[0])))
    newImg = SpecialCrop(size, 1)(newImg)
    newImg = Pad(newDim)(newImg)
    # Resize to output
    v_Img = torch.autograd.Variable(newImg)
    v_Img = torch.unsqueeze(v_Img, 0)
    # newImg = F.upsample_bilinear(v_Img, size=(int(resH), int(resW))).data[0]
    if torch.__version__ == '0.4.0a0+32f3bf7' or torch.__version__ == '0.4.0':
        newImg = F.upsample(v_Img, size=(int(resH), int(resW)),
                            mode='bilinear', align_corners=True).data[0]
    else:
        newImg = F.interpolate(v_Img, size=(int(resH), int(resW)),
                               mode='bilinear', align_corners=True).data[0]
    return newImg

def transformBox(pt, ul, br, inpH, inpW, resH, resW):
    center = torch.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = torch.zeros(2)
    _pt[0] = pt[0] - ul[0]
    _pt[1] = pt[1] - ul[1]
    # Move to center
    _pt[0] = _pt[0] + max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] + max(0, (lenH - 1) / 2 - center[1])
    pt = (_pt * resH) / lenH
    pt[0] = round(float(pt[0]))
    pt[1] = round(float(pt[1]))
    return pt.int()

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def drawGaussian(img, pt, sigma):
    #img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img

def read_image(dirs, data):
    img = cv2.imread(os.path.join(dirs, data['filename']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_box_and_point(data):
    d = data.copy()
    box = [ [i['x'], i['y'], i['width'], i['height']] for i in d['annotations'] if i['class']=='Face' ]
    point = [ [i['x'], i['y']] for i in d['annotations'] if i['class']=='point' ]    
    return box, point

def crop_yolo(img, arr):
    '''arr in format: x, y, width, height'''
    img2 = img.copy()
    return img2[int(arr[1]):int(arr[1])+int(arr[3]), int(arr[0]):int(arr[0])+int(arr[2])]

def rearrange_point(point, num_pose, num_mouse):
    r = np.arange(0, num_pose*num_mouse[0], num_pose)
    point_ = []
    for i in r:
        q = point[i:i+num_pose]
        point_.append(q)
    return point_
    
    
def main_generate_train(train_dir, num_mouse, num_pose, step):
    #step = train_data[50]
    img = read_image(train_dir, step); im = np.transpose(img, (2, 0, 1))/255.0
    box, point = get_box_and_point(step)
    point = rearrange_point(point, num_pose, num_mouse)

    #cropped = [cropBox(to_torch(im), torch.Tensor((i[0], i[1])), 
    #           torch.Tensor((i[0]+i[2], i[1]+i[3])), 320, 256) for i in box]

    #### get cropped training image
    cropped = torch.zeros(num_mouse[0], 3, 320, 256)
    for box_c, i in enumerate(box):
        cropped[box_c] = cropBox(to_torch(im), torch.Tensor((i[0], i[1])), 
                                 torch.Tensor((i[0]+i[2], i[1]+i[3])), 320, 256)

    #### get cropped training label
    img_labels = torch.zeros(num_mouse[0], num_pose, 80, 64)
    setMask = torch.zeros(num_mouse[0], num_pose, 80, 64)
    points_record = np.zeros((num_mouse[0], num_pose, 2))
    for c,i in enumerate(box):
        out = []
        #pp = []
        for j in range(num_pose):
            blank = np.zeros((80, 64))
            part = transformBox(point[c][j], (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), 
                                320, 256, 80, 64)
            part2 = transformBox(point[c][j], (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), 
                                320, 256, 320, 256)
            out.append(drawGaussian(blank, part.numpy(), 1))
            
            points_record[c,j,0] = part.numpy()[0]
            points_record[c,j,1] = part.numpy()[1]
            
        
        out = to_torch(np.array(out))
        img_labels[c] = out
        setMask[c][j].add_(1)

    #print(cropped.shape, img_labels.shape, setMask.shape)
    return cropped, img_labels, setMask, points_record
    

def transform(img):
    #img = np.transpose(img, (2,0,1))
    img = torch.from_numpy(img).float()
    
    img[0].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
    img[1].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
    img[2].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)
    return img.numpy()
    
def augmentation():
    sometimes = lambda aug: iaa.Sometimes(0.25, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.2), # vertical flips
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        sometimes(iaa.OneOf([
                        iaa.GaussianBlur((0, 5.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ])),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 2.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.25*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 3.2), per_channel=0.8),
        
        sometimes(iaa.Affine(
                #scale={"x": (0.6, 0.8), "y": (0.6, 0.8)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                #shear=(-4, 4),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),
        
        sometimes(iaa.Superpixels(
                  p_replace=(0, 1.0),
                  n_segments=(20, 200))),
        
    ], random_order=True) 
    
    return seq
