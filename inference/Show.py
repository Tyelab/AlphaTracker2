'''


'''


import torchvision
import torch

import numpy as np
import os
import cv2
import json



def show_tracked(tracked, vidpath, max_pid_id_setting, start, end, save=False, savepath=''):
    max_pid_id_setting = 2
    vid = cv2.VideoCapture(vidpath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_Frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    keys = list(tracked.keys())
    ids = np.arange(0, max_pid_id_setting) + 1
    index_range = range(start, end)
    
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video=cv2.VideoWriter(savepath, fourcc, 20,(480, 320))
    
    for i in range(0, end):
        ret, frame = vid.read()
        if i in index_range:
            #print(i-start)
            rr = tracked[keys[i-start]]; #print(rr)
            keys2 = list(rr.keys())[1:]

            img = frame.copy()
            cols = {1: (255, 0, 0), 2: (0, 0, 255)}
            for jj in keys2:
                box_position = rr[jj]['box_pose_pos']
                id_ = rr[jj]['new_pid']
                for bp in box_position:
                    cv2.circle(img, (int(bp[0]), int(bp[1])), 15, cols[id_], -1)
                                   
        else:       
            img = frame.copy()
            
        img = cv2.resize(img, (480, 320))
        cv2.imshow("frame", img)
        cv2.waitKey(10)
        
        if save:
            # choose codec according to format needed
            video.write(img)

    
    if save:
        video.release()
        print('confirmed: video saved at: {}'.format(savepath))
    vid.release()
    cv2.destroyAllWindows()
    
        