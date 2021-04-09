import torchvision
import torch
import numpy as np
import cv2
import time
import json
import os
from tqdm import tqdm
from natsort import natsorted
import collections

from . import LoadModels
#from . import tracker
from . import PredictionUtils
from . import TrackerUtils

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

def predict(video_path, nClasses, experiment_name='', size_value=640, img_factor=1, conf=0.05, iou=0.6,
            start=0, end=10000000, detector_input='', estimator_input='', save=False,
            best_or_last='best'):

    ''' Main inference steps to generate un-tracked predictions
    
    Args:
        video_path (str): String of the path for the video to be predicted
        
        detector (PyTorch model): Object detection model, use alphatracker_two.inference.LoadModels.object_detector
        
        estimator (PyTorch model): Pose estimation model, use alphatracker_two.inference.LoadModels.pose_estimator
        
        nClasses (int): number of bodyparts 
        
        start (int): Starting frame to make predictions on
        
        end (int): Last frame to make predictions on, to predict on the complete video, set to a very large number
                
        
    Returns:
        results (list): List of dictionaries that contain un-tracked predictions
        
        runtime_profile (list): List of dictionaries that contain the time profile per frame (in seconds)
    
    '''
       
    if experiment_name and detector_input=='':
        detector_path = os.path.join(experiment_name, 'Weights', 'ObjectDetector')
        detector_files = os.listdir(detector_path); 
        detector_files = natsorted(detector_files); print(detector_files)
        
        
        
        if len(detector_files) > 1:
            detector_files = detector_files[1:]
            detector_full_path = os.path.join(detector_path, detector_files[np.argmax([int(o[3:]) for o in detector_files])], 'weights',
                                              'best.pt')
            
                  
        else:
            detector_full_path = os.path.join(detector_path, 'exp', 'weights', '{}.pt'.format(best_or_last))
            
            
        print("Using object detector: {}".format(detector_full_path))
            
            
    if experiment_name and estimator_input=='':
        estimator_path = os.path.join(experiment_name, 'Weights', 'PoseEstimator')
        estimator_files = os.listdir(estimator_path)
        estimator_files = natsorted(estimator_files); print(estimator_files)
        
        
        if len(estimator_files) > 1:
            estimator_files = estimator_files[1:]; print(estimator_files)
            estimator_full_path = os.path.join(estimator_path, estimator_files[np.argmax([int(o[3:]) for o in estimator_files])])
            estimator_type = os.listdir(estimator_full_path)[0].split('.')[0]
            
            estimator_full_path = os.path.join(estimator_full_path, '{}.best.pkl'.format(estimator_type))
            
                                               
        else:
            estimator_full_path = os.path.join(estimator_path, 'exp')
            estimator_type = os.listdir(estimator_full_path)[0].split('.')[0]
            
            estimator_full_path = os.path.join(estimator_full_path, '{}.{}.pkl'.format(estimator_type, best_or_last))
            
        print("Using pose estimator: {}".format(estimator_full_path))
            
        
        detector = LoadModels.object_detector(detector_full_path)
        estimator = LoadModels.pose_estimator(nClasses, estimator_type, estimator_full_path)
        
        
    if detector_input:
        if type(detector_input) == str:
            detector = LoadModels.object_detector(detector_input)
            
        else:
            detector = detector_input
        
    if estimator_input:
        if type(estimator_input) == str:
            estimator_type = estimator_input.split('.')[0].split(os.sep)[-1]
            estimator = LoadModels.pose_estimator(nClasses, estimator_type, estimator_input)
            
        else:
            estimator = estimator_input
            
    
     
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_Frames = vid.get(cv2.CAP_PROP_FRAME_COUNT); print(total_Frames)
    
    inputResH = int(320/img_factor)
    inputResW = int(256/img_factor)
    outputResH = int(80/img_factor)
    outputResW = int(64/img_factor)
    results = []
    
    detector.conf = conf
    detector.iou = iou
    print('det conf is {}'.format(detector.conf))
    print('det iou is {}'.format(detector.iou))
    
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
    
    
    # start reading video 
    for all_frames in tqdm(range(0, int(end_range))):
        ret, frame = vid.read()
        
        if (all_frames in total_range) == True:
            t_start = time.time()
            #ret, frame = vid.read()
            #to_plot = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            start_time = getTime();
            #### get detections and plot
            #xyxy = detector(frame, size=320).xyxy
            #xyxy = detector(frame, size=640).xyxy
            xyxy = detector(frame, size=size_value, augment=False).xyxy; #print(all_frames, len(xyxy[0]))
            #if len(xyxy[0] > 0):
            #    for d in xyxy[0]:
            #        cv2.rectangle(to_plot, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), 
            #                      (255,0,0), 2)
                                                                   
            inp = PredictionUtils.im_to_torch(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inps = torch.zeros(xyxy[0].size(0), 3, inputResH, inputResW)
            pt1 = torch.zeros(xyxy[0].size(0), 2)
            pt2 = torch.zeros(xyxy[0].size(0), 2)
            boxes = torch.from_numpy(np.array([i[0:4].tolist() for i in xyxy[0]])); #print('boxes is {}'.format(boxes.shape)) 
            scores = [float(ii[4]) for ii in xyxy[0]]  ## ii[4] x1, y1, x2, y2, score, class; class = ....
            im_name = 'frame_{}'.format(all_frames)
            
            #### get pose estimates if detection was successful, and plot
            if boxes.shape[0] > 0:
                
                ckpt_time, det_time = getTime(start_time)
                runtime_profile['detection_time'].append(det_time)
                inps, pt1, pt2 = PredictionUtils.crop_from_dets(inp, boxes, inps, pt1, pt2, inputResH, inputResW)
                
                if torch.cuda.is_available():
                    #inps = inps.cuda()
                    inps = inps.to('cuda')
                
                with torch.no_grad():
                    hm = estimator(inps);
                    

                
                preds_hm, preds_img, preds_scores = PredictionUtils.getPrediction(hm.cpu(), pt1, pt2, 
                                                                                  inputResH, inputResW, 
                                                                                  outputResH, outputResW, nClasses);
                
                #print(preds_img)
                
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pose_estimation_time'].append(pose_time)
                
                #for rr in preds_img: 
                #    rr = rr.tolist()
                #    for jj in rr:
                #        cv2.circle(to_plot, (int(jj[0]), int(jj[1])), 5, (255, 0, 255), -1)            

                result = PredictionUtils.pose_nms(boxes, torch.from_numpy(np.array(scores)), preds_img, preds_scores)


                if len(result) == 0:
                    results.append({'imgname': im_name,
                                'result': [], 
                                'boxes': []})
                
                else:       
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
                
            if len(results[all_frames-start]['result']) == 0:
                results[all_frames-start]['result'] = results[all_frames-start-1]['result']
                results[all_frames-start]['boxes'] = results[all_frames-start-1]['boxes']
                
                
                
            #print(all_frames-start)
            
            #### display image on screen
            #cv2.imshow("frame", to_plot)
            #cv2.waitKey(1)
            
            
            
        
    
    
    vid.release()
    cv2.destroyAllWindows()
    
    return results, runtime_profile, hm
    
    
def track(results, num_obj, experiment_name='', save=False, savepath=''):

    ''' Perform tracking across frames on predicted pose estimates 
    
    Args:
        results (list): Pose estimates, Use output of alphatracker_two.Inference.Infer.predict 
        
        num_obj (int): number of objects in the video
        
        experiment_name (str): String value for the path to the experiment 
        
        save (bool): If True, output json file will be saved in 'experiment_name/Results'
    
    
    Returns:
        tracks (dict): Contains frame-by-frame tracked pose estimates
        
        no_det_frames (list): List of frame IDs where no objects where detected

    '''
    
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
    for iii in range(0, len(results[:-1])):
    
        one = results[iii]
        two = results[1:][iii]
        #print(one['imgname'], one['result'])
        
        if len (one['result'] ) == 0:
            no_det_frames.append(one['imgname'])
            continue
            
        if len (two['result']) == 0:
            no_det_frames.append(two['imgname'])
            continue
            
     
   
        fovis1 = TrackerUtils.convert_tracker(one)
        fovis2 = TrackerUtils.convert_tracker(two)
        
       
        

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
        
    no_det_frames = natsorted(np.unique(no_det_frames).tolist(), lambda x: x.lower())
    
    
    if len(no_det_frames) > 0:
        for no_len in no_det_frames:
            tracks[no_len] = {'num_boxes': 0}
            
    tracks = dict(natsorted(tracks.items()))
    #tracks = collections.OrderedDict(sorted(tracks.items()))
    
    if save:
        #if experiment_name:
        #    full_save_path = os.path.join(experiment_name)
        #    save_path = os.path.join(experiment_name, 'Results', 'tracked.json')
        #    with open(save_path, 'w') as json_file:
        #        json_file.write(json.dumps(tracks, indent=4))
                
        #else:
        #    print("Nothing was saveed, must provide a value for 'experiment_name'!")
        import numpy
        import json
        from json import JSONEncoder
        class NumpyArrayEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, numpy.ndarray):
                    return obj.tolist()
                return JSONEncoder.default(self, obj)
                
        if savepath:
            if os.path.exists(savepath):
                print('Filepath exists, try a new name!')
            else:
                with open(savepath, 'w') as f:
                    json.dump(tracks, f, cls=NumpyArrayEncoder, indent=4)
                
        else:
            
            i = 0
            path_to_save = os.path.join(experiment_name, 'Results', 'tracked_{}.json'.format(i) )
            while os.path.exists(path_to_save):
                i += 1
                path_to_save = os.path.join(experiment_name, 'Results', 'tracked_{}.json'.format(i) )
            else:
                with open(path_to_save, 'w') as f:
                    json.dump(tracks, f, cls=NumpyArrayEncoder, indent=4)
                print('saved')
                
        
    
    return tracks, no_det_frames
    
    
    
    
    
def predict_and_track(video_path, nClasses, num_obj, experiment_name='', detector_input='', estimator_input='', best_or_last='best',
                      conf=0.05, iou=0.6, size_value=640, img_factor=1, start=0, end=10000000, save=True):

    results, runtime, hm = predict(video_path, nClasses, experiment_name, size_value=size_value, img_factor=img_factor, 
                                   conf=conf, iou=iou, start=start, end=end, detector_input=detector_input, estimator_input=estimator_input,
                                   best_or_last=best_or_last)
            
    tracked, no_dets = track(results, num_obj, save=save, experiment_name = experiment_name)
    
    return tracked, no_dets
