''' Used to generate visualizations of tracked outputs

'''

import torchvision
import torch

import numpy as np
import os
import cv2
import json



#def show_tracked(tracked, vidpath, max_pid_id_setting, start, end, save=False, savepath='', fps=20, size=(480,320), 
#                 marker_size=15, line_size=3, playback_speed=10, skeleton=False):

def show_tracked(tracked_, vidpath, experiment_name, start=0, end=100000, save=False, savepath='', fps=20, size=(480,320), 
                 marker_size=15, line_size=3, playback_speed=10, skeleton=False):

    """Takes tracked outputs and displays them in video format

    Args:
        tracked (str or dict): Tracked outputs received from alphtracker_two.inference.Infer.tracked
        
        vidpath (str): The location of the input video 
        
        experiment_name (str): String for the path to the experiment name
        
        start (int): Starting frame to tracked
        
        end (int): Last frame to track, to track the complete video, set this value to a very large number
        
        save (bool): A flag used to indicate whether to save a labeled video 
        
        savepath (str): The location where the video with labels should be saved, only relevant is save=True
        
        fps (int): Frames per second for the labeled video that is generated, only relevant if save=True
        
        size (tuple): Size of the output video, Width by Height, only relevant if save=True
        
        marker_size (int): Size of the marker plotted on the animal
        
        line_size (int): Size of the line for the pose-pairs
        
        playback_speed (int): Time (ms) between frames to be displayed
        
        skeleton (bool): Bool indicating whether skeleton should be plotted 
        

    Returns:
        No returns, if save=True, then a video with labels is saved at the file location (savepath)
        
    """
    
    if type(tracked_) == str:
        with open(tracked_, 'r') as f:
            tracked = json.load(f)
            
    else:
        tracked = tracked_
       
    end = int(len(list(tracked.keys())))
    file = open(os.path.join(experiment_name, 'setting.py'), 'r')
    num_parts = int(file.readlines()[4].split('\n')[0])
    
    file = open(os.path.join(experiment_name, 'setting.py'), 'r')
    num_obj = int(file.readlines()[3].split('\n')[0])
            
    max_pid_id_setting = num_obj
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
        if savepath:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video=cv2.VideoWriter(savepath, fourcc, fps, size)
        else:
            savepath = os.path.join(experiment_name, 'Results', 'tracked_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video=cv2.VideoWriter(savepath, fourcc, fps, size)
    
    for i in range(0, int(end_range)):
        ret, frame = vid.read()
        if i in index_range:
            #print(i-start)
            rr = tracked[keys[i-start]]; #print(rr)
            keys2 = list(rr.keys())[1:]
            
            if skeleton:
                img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            else:
                img = frame.copy()

            #img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
            cols = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0,0,255), 3: (255,255,255), 4: (255,0,0), 
                    5: (0, 255, 255), 6: (255, 255, 0), 7: (255,0,255), 8: (50,255,125), 9: (125,125,125)}
            for jj in keys2:
                box_position = rr[jj]['box_pose_pos']
                id_ = rr[jj]['new_pid']
                for ccc, bp in enumerate(box_position):
                    if max_pid_id_setting == 1:
                        cv2.circle(img, (int(bp[0]), int(bp[1])), marker_size, cols[id_], -1) # should be id_
                    else:
                        cv2.circle(img, (int(bp[0]), int(bp[1])), marker_size, cols[id_], -1) # should be cols[id_]
                    
                cake = box_position.copy()
                #cv2.line(img, (int(cake[0][0]), int(cake[0][1])), (int(cake[1][0]), int(cake[1][1])), cols[id_], line_size)
                #cv2.line(img, (int(cake[1][0]), int(cake[1][1])), (int(cake[2][0]), int(cake[2][1])), cols[id_], line_size)
                #cv2.line(img, (int(cake[2][0]), int(cake[2][1])), (int(cake[3][0]), int(cake[3][1])), cols[id_], line_size)
                #cv2.line(img, (int(cake[3][0]), int(cake[3][1])), (int(cake[1][0]), int(cake[1][1])), cols[id_], line_size)
                #cv2.line(img, (int(cake[0][0]), int(cake[0][1])), (int(cake[2][0]), int(cake[2][1])), cols[id_], line_size)
                
                                   
        else:       
            img = frame.copy()
          
        font = cv2.FONT_HERSHEY_SIMPLEX; bottomLeftCornerOfText = (40,80); fontScale = 3
        fontColor = (255,255,255); lineType = 5

        if not skeleton:
            cv2.putText(img,'frame_{}'.format(i), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
        img = cv2.resize(img, size)
        cv2.imshow("frame", img)
        cv2.waitKey(playback_speed)
        
        if save:
            # choose codec according to format needed
            video.write(img)

    
    if save:
        video.release()
        print('confirmed: video saved at: {}'.format(savepath))
    print(img.max())
    vid.release()
    cv2.destroyAllWindows()
    
    if save==False:
        print('Video was not saved')
        print('To save the tracked video with labels, use "save=True" and enter in the desired save location in "savepath"')
    
        
