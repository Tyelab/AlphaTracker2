''' Used to generate visualizations of tracked outputs

'''

import torchvision
import torch

import numpy as np
import os
import cv2
import json



def show_tracked(tracked, vidpath, max_pid_id_setting, start, end, save=False, savepath='', fps=20, size=(480,320)):

    """Takes tracked outputs and displays them in video format

    Args:
        tracked (dict): Tracked outputs received from alphtracker_two.inference.Infer.tracked
        
        vidpath (str): The location of the input video 
        
        max_pid_id_setting (int): Maximum number of objects in the video to be tracked
        
        start (int): Starting frame to tracked
        
        end (int): Last frame to track, to track the complete video, set this value to a very large number
        
        save (bool): A flag used to indicate whether to save a labeled video 
        
        savepath (str): The location where the video with labels should be saved, only relevant is save=True
        
        fps (int): Frames per second for the labeled video that is generated, only relevant if save=True
        
        size (tuple): Size of the output video, Width by Height, only relevant if save=True
        

    Returns:
        No returns, if save=True, then a video with labels is saved at the file location (savepath)
        
    """
    
    max_pid_id_setting = max_pid_id_setting
    vid = cv2.VideoCapture(vidpath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_Frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    keys = list(tracked.keys())
    ids = np.arange(0, max_pid_id_setting) + 1
    index_range = range(start, end)
    
    if total_Frames >= end:
        end_range = end
    else:
        end_range = total_Frames
    
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video=cv2.VideoWriter(savepath, fourcc, fps, size)
    
    for i in range(0, int(end_range)):
        ret, frame = vid.read()
        if i in index_range:
            #print(i-start)
            rr = tracked[keys[i-start]]; #print(rr)
            keys2 = list(rr.keys())[1:]

            img = frame.copy()
            cols = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0,0,0), 3: (255,255,255)}
            for jj in keys2:
                box_position = rr[jj]['box_pose_pos']
                id_ = rr[jj]['new_pid']
                for ccc, bp in enumerate(box_position):
                    cv2.circle(img, (int(bp[0]), int(bp[1])), 15, cols[ccc], -1) # should be cols[id_]
                                   
        else:       
            img = frame.copy()
          
        font = cv2.FONT_HERSHEY_SIMPLEX; bottomLeftCornerOfText = (40,80); fontScale = 3
        fontColor = (255,255,255); lineType = 5

        cv2.putText(img,'frame_{}'.format(i), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
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
    
    if save==False:
        print('Video was not saved')
        print('To save the tracked video with labels, use "save=True" and enter in the desired save location in "savepath"')
    
        