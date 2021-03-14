import torchvision
import torch
import numpy as np
import cv2
import time

from . import LoadModels
from . import tracker
from . import PredictionUtils
from . import TrackerUtils

def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval

def real_time_tracking(detector, estimator, nClasses):
    vid = cv2.VideoCapture(0)
    inputResH = 320
    inputResW = 256
    outputResH = 80
    outputResW = 64
    results = []
    fps_per_frame = []
    
    runtime_profile = {
        'detection_time': [],
        'pose_estimation_time': [],
        'post_processing_time': [],
        'per_frame_time': []
    }

    
    try:
        counter_ = 0
        t0 = time.time()
        #start_time = getTime()
        while 1:
            t_start = time.time()
            ret, frame = vid.read()
            to_plot = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
                                                                                  outputResH, outputResW, nClasses);
                
                
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pose_estimation_time'].append(pose_time)
                
                for rr in preds_img: 
                    rr = rr.tolist()
                    for jj in rr:
                        cv2.circle(to_plot, (int(jj[0]), int(jj[1])), 5, (255, 0, 255), -1)
            
            
                
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

            
            #### display image on screen
            cv2.imshow("frame", to_plot)
            cv2.waitKey(1)
            counter_ += 1
            frame_time = time.time() - t_start
            fps_per_frame.append(1/frame_time)
            #print(1/frame_time)
            
    except:
        #### if user interrupts process from keyboard, close the image window and shut off the webcam streaming
        KeyboardInterrupt
        vid.release()
        cv2.destroyAllWindows()
        
        return results, runtime_profile
        
        
    vid.release()
    cv2.destroyAllWindows()
    
            