import torchvision
import torch
import numpy as np
import cv2
import time
import sys

from . import LoadModels
from . import tracker
from . import PredictionUtils
from . import TrackerUtils
from . import Show
from . import Infer

def getTime(time1=0):

    ''' Get time to run code for optimization
    
    Args:
        time1 (int): Initial time before desired process occurs
        
        
    Returns:
        time (float): Time elapsed since the first process
        
    '''
    
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval


def real_time_tracking(detector, estimator, nClasses, max_pid_id_setting, save=False, savepath='', fps=15, size=(640,480)):

    """ Perform real-time predictions and tracking of pose estimates

    Args:
        detector (PyTorch model): Object detection model, use alphatracker_two.inference.LoadModels.object_detector
        
        estimator (PyTorch model): Pose estimation model, use alphatracker_two.inference.LoadModels.pose_estimator
        
        nClasses (int): number of bodyparts 
        
        max_pid_id_setting (int): Maximum number of objects in the video to be tracked
        
        save (bool): A flag used to indicate whether to save a labeled video 
        
        savepath (str): The location where the video with labels should be saved, only relevant is save=True
        
        fps (int): Frames per second for the labeled video that is generated, only relevant if save=True
        size (tuple): Size of the output video, Width by Height, only relevant if save=True
        
        size (tuple): Size of the output video, Width by Height, only relevant if save=True
        

    Returns:
        results (list): List of dictionaries that contain un-tracked predictions
        
        runtime_profile (list): List of dictionaries that contain the time profile per frame (in seconds)
    
        tracked (dict): Contains frame-by-frame tracked pose estimates
        
    """
    
    vid = cv2.VideoCapture(0)
    inputResH = 320
    inputResW = 256
    outputResH = 80
    outputResW = 64
    results = []
    fps_per_frame = []
    tracked_ = []
    nd = []
    
    runtime_profile = {
        'detection_time': [],
        'pose_estimation_time': [],
        'post_processing_time': [],
        'per_frame_time': []
    }

    
    width = 480.0#vid.get(cv2.CV_CAP_PROP_FRAME_WIDTH)   # float
    height = 320. #vid.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) # float


    #fourcc = cv2.CV_FOURCC(*'X264')
    if save == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(savepath, fourcc, fps, size) 
    
    try:
        counter_ = 0
        cols = {1: (255, 0, 0), 2: (0, 0, 255)}

        t0 = time.time()
        #start_time = getTime()
        while 1:
            t_start = time.time()
            ret, frame = vid.read()
            to_plot = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if save == True:
                out.write(frame)
            
            start_time = getTime()
            #### get detections and plot
            xyxy = detector(frame, size=320).xyxy
            if len(xyxy[0] > 0):
                for d in xyxy[0]:
                    cv2.rectangle(to_plot, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), 
                                  (255,0,0), 2)
                                  
                                  
            inp = PredictionUtils.im_to_torch(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inps = torch.zeros(xyxy[0].size(0), 3, inputResH, inputResW)
            pt1 = torch.zeros(xyxy[0].size(0), 2)
            pt2 = torch.zeros(xyxy[0].size(0), 2)
            boxes = torch.from_numpy(np.array([i[0:4].tolist() for i in xyxy[0]]))
            scores = [float(ii[4]) for ii in xyxy[0]]
            im_name = 'frame_{}'.format(counter_) 
            
            
            
            #### get pose estimates if detection was successful, and plot
            if boxes.shape[0] > 0:
                
                ckpt_time, det_time = getTime(start_time)
                runtime_profile['detection_time'].append(det_time)
                inps, pt1, pt2 = PredictionUtils.crop_from_dets(inp, boxes, inps, pt1, pt2, inputResH, inputResW)
                with torch.no_grad():
                    hm = estimator(inps)
                    
                preds_hm, preds_img, preds_scores = PredictionUtils.getPrediction(hm.cpu(), pt1, pt2, 
                                                                                  inputResH, inputResW, 
                                                                                  outputResH, outputResW, nClasses); #print(preds_hm)
                
                
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pose_estimation_time'].append(pose_time)
                
                #for rr in preds_img: 
                #    rr = rr.tolist()
                #    for jj in rr:
                #        cv2.circle(to_plot, (int(jj[0]), int(jj[1])), 5, (255, 0, 255), -1)
            
            
                
                result = PredictionUtils.pose_nms(boxes, torch.from_numpy(np.array(scores)), preds_img, preds_scores)
                result = {
                            'imgname': im_name,
                            'result': result,
                            'boxes':boxes
                        }
            
                results.append(result)
                
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['post_processing_time'].append(post_time)
                
                runtime_profile['per_frame_time'].append((time.time()-t_start))
                
                
                
            else:
                results.append({'imgname': im_name,
                                'result': [], 
                                'boxes': []})
                                
                runtime_profile['detection_time'].append(np.nan)
                runtime_profile['pose_estimation_time'].append(np.nan)
                runtime_profile['post_processing_time'].append(np.nan)
                runtime_profile['per_frame_time'].append((time.time()-t_start))
                
            if counter_ > 0:
                tracked1, not_detected1 = Infer.track(results, max_pid_id_setting)
                #print(tracked1)
                if counter_ >= 1:
                    key_current = 'frame_{}'.format(counter_-1); 
                    obj = tracked1[key_current]; 
                    keys2 = list(obj.keys())[1:]
                    if len(keys2) > 0:
                        cols = {1: (255, 0, 0), 2: (0, 0, 255)}
                        for keys_in_2 in keys2:
                            pid_for_this_one = obj[keys_in_2]['new_pid']
                            for bp in obj[keys_in_2]['box_pose_pos']:
                                cv2.circle(to_plot, (int(bp[0]), int(bp[1])), 15, cols[pid_for_this_one], -1)
                            
                #print('first')
            
            #if counter_ >= 20:
            #    tracked, not_detected = Infer.track(results[-3:], max_pid_id_setting);
            #    key_current = 'frame_{}'.format(counter_-1);
            #    obj = tracked[key_current]; 
            #    keys2 = list(obj.keys())[1:]
            #    #if len(keys2) > 0:
            #    cols = {1: (255, 0, 0), 2: (0, 0, 255)}
            #    for keys_in_2 in keys2:
            #        pid_for_this_one = obj[keys_in_2]['new_pid']
            #        for bp in obj[keys_in_2]['box_pose_pos']:
            #            cv2.circle(to_plot, (int(bp[0]), int(bp[1])), 15, cols[pid_for_this_one], -1)

            #    tracked_.append(tracked)
            #    nd.append(not_detected)
                #print('second')
            #### display image on screen
            cv2.imshow("frame", to_plot)
            cv2.waitKey(1)
            counter_ += 1
            frame_time = time.time() - t_start
            fps_per_frame.append(1/frame_time)
            #print(1/frame_time)
        
            
    except:
        #print(e)
        
        #return tracked1, tracked_
        
        #### if user interrupts process from keyboard, close the image window and shut off the webcam streaming
        ## very hacky way to save memory when tallying results
        if counter_ >= 20:
            yy = {}
            for o in range(len(tracked_)):
                obj = tracked_[o]
                if len(obj) == 1:
                    continue
                elif len(obj) == 2:
                   yy[list(obj.keys())[1]] = obj[list(tracked_[o].keys())[1]]
                   
                elif len(obj) == 3:
                    if obj[list(obj.keys())[0]]['num_boxes'] == 0:
                        yy[list(obj.keys())[1]] = obj[list(tracked_[o].keys())[1]]
                        
                    else:
                        yy[list(obj.keys())[2]] = obj[list(tracked_[o].keys())[2]]
            
            tracked1.update(yy)
        
        
        KeyboardInterrupt
        vid.release()
        if save == True:
            out.release()
        cv2.destroyAllWindows()
       
        return results, runtime_profile, tracked1, 
        
        
    vid.release()
    if save == True:
        out.release()
    cv2.destroyAllWindows()
    
            