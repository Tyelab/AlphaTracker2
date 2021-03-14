import torchvision
import torch
import numpy as np
import cv2
import time
import json
import os
from tqdm.notebook import tqdm


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

def predict(video_path, detector, estimator, nClasses, start, end):
    
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_Frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    inputResH = 320
    inputResW = 256
    outputResH = 80
    outputResW = 64
    results = []
    
    runtime_profile = {
        'detection_time': [],
        'pose_estimation_time': [],
        'post_processing_time': [],
        'per_frame_time': []
    }

    #start_time = getTime()
    total_range = range(start, end)
    
    if total_Frames >= end:
        end_range = end
    else:
        end_range = total_Frames
    
    
     
    for all_frames in tqdm(range(0, int(end_range))):
        ret, frame = vid.read()
        
    
        if (all_frames in total_range) == True:
            t_start = time.time()
            #ret, frame = vid.read()
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
            im_name = 'frame_{}'.format(all_frames)           
            
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
            #cv2.imshow("frame", to_plot)
            #cv2.waitKey(1)
            
        
    
    
    vid.release()
    cv2.destroyAllWindows()
    
    return results, runtime_profile
    
    
def track(results, num_obj):
    
    idxss = []
    notracks1 = {}; notracks2 = {}
    tracks = {}
    no_det_frames = []
    num_persons = 0
    max_pid_id = 0

    max_pid_id_setting = num_obj

    weights = [0, 6, 0, 0, 0, 0]
    weights_fff = weights
    match = 0

    idx = 0
    for iii, (one, two) in enumerate(zip(results, results[1:])):
        fovis1 = TrackerUtils.convert_tracker(one)
        fovis2 = TrackerUtils.convert_tracker(two)
        
        if len( one['result'] ) == 0:
            print("warning: detected 0 objects in current frame")
            no_det_frames.append(one['imgname'])
            #idd += 1
            continue
            
        if len( two['result'] ) == 0:
            print("warning: detected 0 objects in next frame")
            no_det_frames.append(two['imgname'])
            #idd += 1
            continue

        #print(fovis2)
        imgpath1 = list(fovis1.keys())[0]
        imgpath2 = list(fovis2.keys())[0]
        idxss.append(imgpath1)
        
        not_t1 = TrackerUtils.toFixNum_notrack2(fovis1.copy(), fovis1.copy().keys(), max_pid_id_setting)
        not_t2 = TrackerUtils.toFixNum_notrack2(fovis2.copy(), fovis2.copy().keys(), max_pid_id_setting)
        
        notracks1[imgpath1] = not_t1[imgpath1]
        notracks2[imgpath2] = not_t2[imgpath2]
        
        
        track1 = {}; track2 = {}
        img_name1 = imgpath1; frame_name1 = imgpath1
        img_name2 = imgpath2; frame_name2 = imgpath2
        
        notrack1 = not_t1.copy()
        notrack2 = not_t2.copy()
        #print(idx-idd)
        
        track1[img_name1] = {'num_boxes':len(notrack1[img_name1])}
        for bid in range(len(notrack1[img_name1])):
            track1[img_name1][bid+1] = {}
            track1[img_name1][bid+1]['box_score'] = notrack1[img_name1][bid]['scores']
            track1[img_name1][bid+1]['box_pos'] =  [ int(notrack1[img_name1][bid]['box'][0]),\
                                                   int(notrack1[img_name1][bid]['box'][2]),\
                                                   int(notrack1[img_name1][bid]['box'][1]),\
                                                   int(notrack1[img_name1][bid]['box'][3])]
            track1[img_name1][bid+1]['box_pose_pos'] = np.array(notrack1[img_name1][bid]['keypoints']).reshape(-1,3)[:,0:2]
            track1[img_name1][bid+1]['box_pose_score'] = np.array(notrack1[img_name1][bid]['keypoints']).reshape(-1,3)[:,-1]
            
        track2[img_name2] = {'num_boxes':len(notrack2[img_name2])}
        for bid in range(len(notrack2[img_name2])):
            track2[img_name2][bid+1] = {}
            track2[img_name2][bid+1]['box_score'] = notrack2[img_name2][bid]['scores']
            track2[img_name2][bid+1]['box_pos'] =  [ int(notrack2[img_name2][bid]['box'][0]),\
                                                   int(notrack2[img_name2][bid]['box'][2]),\
                                                   int(notrack2[img_name2][bid]['box'][1]),\
                                                   int(notrack2[img_name2][bid]['box'][3])]
            track2[img_name2][bid+1]['box_pose_pos'] = np.array(notrack2[img_name2][bid]['keypoints']).reshape(-1,3)[:,0:2]
            track2[img_name2][bid+1]['box_pose_score'] = np.array(notrack2[img_name2][bid]['keypoints']).reshape(-1,3)[:,-1]
            
        
        if idx == 0:
            for pid in range(1, track1[frame_name1]['num_boxes']+1):
                # print('!!!!!!!!!!!in')
                track1[frame_name1][pid]['new_pid'] = pid
                track1[frame_name1][pid]['match_score'] = 0
                
            tracks[img_name1] = track1[img_name1]
            tracks[img_name2] = track2[img_name2]
            
        else:
            
            tracks[img_name2] = track2[img_name2]
        
        
        
        max_pid_id = max(max_pid_id, track1[frame_name1]['num_boxes'])
        
        if track2[frame_name2]['num_boxes'] == 0:
            track2 = copy.deepcopy(track1)
            continue
            
            
        #### stacking and matching
        #print(list(track1.keys()))
        
        # maybe something weird here!
        cur_all_pids, cur_all_pids_fff = TrackerUtils.stack_all_pids(tracks, list(tracks.keys()), idx, max_pid_id, 100)
        #cur_all_pids = [track1[frame_name1][i] for i in range(1, len(track1[frame_name1]))]
        #cur_all_pids_fff = [True, True]
        stack_time = time.time()
        
        
        match_indexes, match_scores = TrackerUtils.best_matching_hungarian_noORB(
            None, cur_all_pids, cur_all_pids_fff, track2[frame_name2], weights, weights_fff, 7, 30)
        
        
        
        
        #### post-matching assignment
        pid_remain = [i+1 for i in range(max_pid_id_setting)]
        pid_conflict = []
        pid2s_checked = []
        pid1s_checked = []
        
        
        for pid1, pid2 in match_indexes:
            if match_scores[pid1][pid2] > 0:

                if max_pid_id_setting!=-1:
                    if pid2 in pid2s_checked or pid1 in pid1s_checked:
                        pid_conflict.append([pid1,pid2])
                        continue
                    else:
                        pid_remain.remove(cur_all_pids[pid1]['new_pid'])
                        pid2s_checked.append(pid2)
                        pid1s_checked.append(pid1)

                tracks[frame_name2][pid2+1]['new_pid'] = cur_all_pids[pid1]['new_pid']
                # max_pid_id = max(max_pid_id, track[next_frame_name][pid2+1]['new_pid'])
                tracks[frame_name2][pid2+1]['match_score'] = match_scores[pid1][pid2]
                    



                # print(pid2+1,track[next_frame_name][pid2+1])

                if tracks[frame_name2][pid2+1]['new_pid']>max_pid_id:
                    print('tracking warning!!\n track[next_frame_name][pid2+1][\'new_pid\']>max_pid_id:',next_frame_name,track[next_frame_name][pid2+1]['new_pid'],max_pid_id)
                
                # print(pid1,pid2,pid_remain,pid_conflict)
                
                
                
        for next_pid in range(1, tracks[frame_name2]['num_boxes'] + 1):
            if 'new_pid' not in tracks[frame_name2][next_pid]:
                if max_pid_id_setting!=-1:
                    if len(pid_remain)!=0:
                        tracks[frame_name2][next_pid]['new_pid'] = pid_remain[0]
                        del pid_remain[0]
                else:
                    max_pid_id += 1
                    tracks[frame_name2][next_pid]['new_pid'] = max_pid_id
                    tracks[frame_name2][next_pid]['match_score'] = 0

        idx += 1
        
    
    return tracks, no_det_frames
