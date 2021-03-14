import torchvision
import torch

import numpy as np
import os
import json

import numpy as np
import cv2 as cv
import os
import json
import copy
import heapq
from munkres import Munkres, print_matrix
from PIL import Image
from tqdm import tqdm
import cv2


def convert_tracker(results):
    arr = results.copy()
    keypoints_and_scores_and_boxes = {}; keypoints_and_scores_and_boxes['keypoints'] = []
    fovis = {}

    for count, a in enumerate(arr['result']):
        keypoints_and_scores_and_boxes = {}; keypoints_and_scores_and_boxes['keypoints'] = []
        kps = a['keypoints'].numpy()
        kp_score = a['kp_score'].numpy()
        scores = float(a['proposal_score'])
        box = a['box']
        keypoints_and_scores_and_boxes['keypoints'] = np.concatenate(( kps, kp_score ), 1).reshape(-1).tolist()
        keypoints_and_scores_and_boxes['scores'] = scores
        keypoints_and_scores_and_boxes['box'] = box.tolist()

        if count == 0:
            fovis[arr['imgname']] = []
            fovis[arr['imgname']].append(keypoints_and_scores_and_boxes)
        else:
            fovis[arr['imgname']].append(keypoints_and_scores_and_boxes)
            
    return fovis
    
    
    
#### #### #### #### # coding: utf-8

'''
File: utils.py
Project: AlphaPose
File Created: Thursday, 1st March 2018 5:32:34 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 20th March 2018 1:18:17 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

# keypoint penalty weight
delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891, 0.01402144, \
                    0.03909642, 0.03686941, 0.01981803, 0.03843971, 0.03412318, 0.02415081, \
                    0.01291456, 0.01236173,0.01291456, 0.01236173])

track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

# get expand bbox surrounding single person's keypoints
def get_box(pose, imgpath):

    pose = np.array(pose).reshape(-1,3)
    xmin = np.min(pose[:,0])
    xmax = np.max(pose[:,0])
    ymin = np.min(pose[:,1])
    ymax = np.max(pose[:,1])
    
    img_height, img_width, _ = cv.imread(imgpath).shape

    return expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height)


# expand bbox for containing more background
def expand_bbox(left, right, top, bottom, img_width, img_height):

    width = right - left
    height = bottom - top
    ratio = 0.1 # expand ratio
    new_left = np.clip(left - ratio * width, 0, img_width)
    new_right = np.clip(right + ratio * width, 0, img_width)
    new_top = np.clip(top - ratio * height, 0, img_height)
    new_bottom = np.clip(bottom + ratio * height, 0, img_height)

    return [int(new_left), int(new_right), int(new_top), int(new_bottom)]

# calculate final matching grade
def cal_grade(l, w):
    return sum(np.array(l)*np.array(w))

# calculate IoU of two boxes(thanks @ZongweiZhou1)
def cal_bbox_iou(boxA, boxB): 

    xA = max(boxA[0], boxB[0]) #xmin
    yA = max(boxA[2], boxB[2]) #ymin
    xB = min(boxA[1], boxB[1]) #xmax
    yB = min(boxA[3], boxB[3]) #ymax

    if xA < xB and yA < yB: 
        interArea = (xB - xA + 1) * (yB - yA + 1) 
        boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1) 
        boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1) 
        iou = interArea / float(boxAArea + boxBArea - interArea+0.00001) 
    else: 
        iou=0.0

    return iou

# stack all already tracked people's info together(thanks @ZongweiZhou1)
def stack_all_pids(track_vid, frame_list, idxs, max_pid_id, link_len):
    
    #track_vid contains track_vid[<=idx]
    all_pids_info = []
    all_pids_fff = [] # boolean list, 'fff' means From Former Frame
    all_pids_ids = [(item+1) for item in range(max_pid_id)]
    
    for idx in np.arange(idxs,max(idxs-link_len,-1),-1):
        # print('!!!',track_vid[frame_list[idx]])
        for pid in range(1, track_vid[frame_list[idx]]['num_boxes']+1):
            if len(all_pids_ids) == 0:
                return all_pids_info, all_pids_fff
            elif track_vid[frame_list[idx]][pid]['new_pid'] in all_pids_ids:
                all_pids_ids.remove(track_vid[frame_list[idx]][pid]['new_pid'])
                all_pids_info.append(track_vid[frame_list[idx]][pid])
                if idx == idxs:
                    all_pids_fff.append(True)
                else:
                    all_pids_fff.append(False)
    return all_pids_info, all_pids_fff


    # calculate general Pose IoU(only consider top NUM matched keypoints)
def cal_pose_iou(pose1_box,pose2_box, num,mag):
    
    pose_iou = []
    for row in range(len(pose1_box)):
        x1,y1 = pose1_box[row]
        x2,y2 = pose2_box[row]
        box1 = [x1-mag,x1+mag,y1-mag,y1+mag]
        box2 = [x2-mag,x2+mag,y2-mag,y2+mag]
        pose_iou.append(cal_bbox_iou(box1,box2))

    return np.mean(heapq.nlargest(num, pose_iou))
        
def select_max(cost_matrix):
    cost_matrix_copy = copy.deepcopy(cost_matrix)
    selectIdx = []
    for ii in range(cost_matrix_copy.shape[1]):
        xs,ys = np.where(cost_matrix_copy==np.max(cost_matrix_copy))
        selectIdx.append((xs[0],ys[0]))
        cost_matrix_copy[xs[0],:] = 0
        cost_matrix_copy[:,ys[0]] = 0

    return selectIdx
    

def best_matching_hungarian_noORB(all_cors, all_pids_info, all_pids_fff, track_vid_next_fid, weights, weights_fff, num, mag):
    
    # x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    all_grades_details = []
    all_grades = []
    
    box1_num = len(all_pids_info)
    box2_num = track_vid_next_fid['num_boxes']
    cost_matrix = np.zeros((box1_num, box2_num))

    for pid1 in range(box1_num):
        box1_pos = all_pids_info[pid1]['box_pos']
        # box1_region_ids = find_region_cors_last(box1_pos, all_cors)
        box1_score = all_pids_info[pid1]['box_score']
        box1_pose = all_pids_info[pid1]['box_pose_pos']
        box1_fff = all_pids_fff[pid1]

        for pid2 in range(1, track_vid_next_fid['num_boxes'] + 1):
            box2_pos = track_vid_next_fid[pid2]['box_pos']
            # box2_region_ids = find_region_cors_next(box2_pos, all_cors)
            box2_score = track_vid_next_fid[pid2]['box_score']
            box2_pose = track_vid_next_fid[pid2]['box_pose_pos']
                        
            # inter = box1_region_ids & box2_region_ids
            # union = box1_region_ids | box2_region_ids
            # dm_iou = len(inter) / (len(union) + 0.00001)
            dm_iou = 0
            box_iou = cal_bbox_iou(box1_pos, box2_pos)
            # pose_iou_dm = cal_pose_iou_dm(all_cors, box1_pose, box2_pose, num,mag)
            pose_iou_dm = 0
            pose_iou = cal_pose_iou(box1_pose, box2_pose,num,mag)
            if box1_fff:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights)
            else:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights_fff)
                
            cost_matrix[pid1, pid2 - 1] = grade
    indexes = select_max(cost_matrix)
    # m = Munkres()
    # indexes = m.compute((-np.array(cost_matrix)).tolist())

    return indexes, cost_matrix

    
def toFixNum_notrack(track_in,frame_list, max_pid_id_setting):
    for idx, frame_name in enumerate(tqdm(frame_list)):
        if len(track_in[frame_name])<=max_pid_id_setting:
            continue
        ## make the first max_pid_id_setting persons  the person with max pose score.
        current_select_score = []
        current_select_score_pid_dict = {}
        for pid in range(max_pid_id_setting):
            current_select_score.append(track_in[frame_name][pid]['scores'])
            current_select_score_pid_dict[track_in[frame_name][pid]['scores']] = pid

        for pid in range(max_pid_id_setting, len(track_in[frame_name])):
            current_min_score = min(current_select_score)
            if track_in[frame_name][pid]['scores'] > min(current_select_score):
                min_score_pid = current_select_score_pid_dict[min(current_select_score)]
                track_in[frame_name][min_score_pid] = track_in[frame_name][pid]
                current_select_score_pid_dict[track_in[frame_name][pid]['scores']] = min_score_pid
                current_select_score.remove(min(current_select_score))
                current_select_score.append(track_in[frame_name][pid]['scores'])
                # track_in[frame_name][pid] = None

        track_in[frame_name] = track_in[frame_name][:max_pid_id_setting]

        # track_in[frame_name]['num_boxes'] = max_pid_id_setting

    return track_in


def toFixNum_notrack2(track_in,frame_list, max_pid_id_setting):
    for idx, frame_name in enumerate(frame_list):
        if len(track_in[frame_name])<=max_pid_id_setting:
            continue
        ## make the first max_pid_id_setting persons  the person with max pose score.
        current_select_score = []
        current_select_score_pid_dict = {}
        for pid in range(max_pid_id_setting):
            current_select_score.append(track_in[frame_name][pid]['scores'])
            current_select_score_pid_dict[track_in[frame_name][pid]['scores']] = pid

        for pid in range(max_pid_id_setting, len(track_in[frame_name])):
            current_min_score = min(current_select_score)
            if track_in[frame_name][pid]['scores'] > min(current_select_score):
                min_score_pid = current_select_score_pid_dict[min(current_select_score)]
                track_in[frame_name][min_score_pid] = track_in[frame_name][pid]
                current_select_score_pid_dict[track_in[frame_name][pid]['scores']] = min_score_pid
                current_select_score.remove(min(current_select_score))
                current_select_score.append(track_in[frame_name][pid]['scores'])
                # track_in[frame_name][pid] = None

        track_in[frame_name] = track_in[frame_name][:max_pid_id_setting]

        # track_in[frame_name]['num_boxes'] = max_pid_id_setting

    return track_in
    
    
    