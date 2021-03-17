import torch
import torch.nn.functional as F
from torchsample.transforms import SpecialCrop, Pad
import numpy as np
import cv2


delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
gamma_avg = 22.48/17
scoreThreds = 0.3
# scoreThreds = 0
matchThreds = 5
matchThreds_ratio = 0.6
areaThres = 0#40 * 40.5
alpha = 0.1
#pool = ThreadPool(4)



def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


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


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


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


def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW, num_pos):
    '''
    pt:     [n, 17, 2]
    ul:     [n, 2]
    br:     [n, 2]
    '''
    center = (br - 1 - ul) / 2

    size = br - ul
    size[:, 0] *= (inpH / inpW)

    lenH, _ = torch.max(size, dim=1)   # [n,]
    lenW = lenH * (inpW / inpH)

    _pt = (pt * lenH[:, np.newaxis, np.newaxis]) / resH
    _pt[:, :, 0] = _pt[:, :, 0] - ((lenW[:, np.newaxis].repeat(1, num_pos) - 1) /
                                   2 - center[:, 0].unsqueeze(-1).repeat(1, num_pos)).clamp(min=0)
    _pt[:, :, 1] = _pt[:, :, 1] - ((lenH[:, np.newaxis].repeat(1, num_pos) - 1) /
                                   2 - center[:, 1].unsqueeze(-1).repeat(1, num_pos)).clamp(min=0)

    new_point = torch.zeros(pt.size())
    new_point[:, :, 0] = _pt[:, :, 0] + ul[:, 0].unsqueeze(-1).repeat(1, num_pos)
    new_point[:, :, 1] = _pt[:, :, 1] + ul[:, 1].unsqueeze(-1).repeat(1, num_pos)
    return new_point
    
    
    
    
    
    
    
    
    
    
    
    
def crop_from_dets(img, boxes, inps, pt1, pt2, inputResH, inputResW):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img, upLeft, bottomRight, inputResH, inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2



def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW, num_pos):
    '''
    Get keypoint location from heatmaps
    '''
    outputResH = resH
    outputResW = resW
    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(round(float(preds[i][j][1])))
            if 0 < pX < outputResW - 1 and 0 < pY < outputResH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                preds[i][j] += diff.sign() * 0.25
    preds += 0.2

    preds_tf = torch.zeros(preds.size())
    
    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW, num_pos)

    return preds, preds_tf, maxval
    
    
    
    
    
    
    
    
    
    
def pose_nms(bboxes, bbox_scores, pose_preds, pose_scores):
    '''
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n,)
    pose_preds:     pose locations list (n, 17, 2)
    pose_scores:    pose scores list    (n, 17, 1)
    '''
    #global ori_pose_preds, ori_pose_scores, ref_dists

    pose_scores[pose_scores == 0] = 1e-5

    final_result = []

    ori_bbox = bboxes.clone()
    ori_bbox_scores = bbox_scores.clone()
    ori_pose_preds = pose_preds.clone()
    ori_pose_scores = pose_scores.clone()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(dim=1)
    npose = pose_preds.shape[1]

    human_ids = np.arange(nsamples)
    # print('npose_nms.py:',bboxes.shape,human_ids,nsamples)
    # Do pPose-NMS
    pick = []
    merge_ids = []
    # while(human_scores.shape[0] != 0):
    while(human_ids.shape[0] != 0):
        # Pick the one with highest score
        pick_id = torch.argmax(human_scores)
        # print('npose_nms.py:',bboxes.shape,nsamples,human_ids,pick_id,human_scores.shape)

        pick.append(human_ids[pick_id])
        # num_visPart = torch.sum(pose_scores[pick_id] > 0.2)

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[pick_id]]
        simi = get_parametric_distance(pick_id, pose_preds, pose_scores, ref_dist)
        num_match_keypoints = PCK_match(pose_preds[pick_id], pose_preds, ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        # delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[(simi > gamma) | (num_match_keypoints >= matchThreds)]
        # delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[(simi > gamma) | (num_match_keypoints >= int(npose*matchThreds_ratio))]
        delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[((simi)/npose > gamma) | (num_match_keypoints >= int(npose*matchThreds_ratio))]

        # print('pPose-NMS.py:delete_ids:',delete_ids)
        # print('pPose-NMS.py:pick_id:',pick_id)
        # print('pPose-NMS.py:merge_ids:',merge_ids)
        if delete_ids.shape[0] == 0:
            delete_ids = pick_id
        #else:
        #    delete_ids = torch.from_numpy(delete_ids)

        merge_ids.append(human_ids[delete_ids])
        pose_preds = np.delete(pose_preds, delete_ids, axis=0)
        pose_scores = np.delete(pose_scores, delete_ids, axis=0)
        human_ids = np.delete(human_ids, delete_ids)
        human_scores = np.delete(human_scores, delete_ids, axis=0)
        bbox_scores = np.delete(bbox_scores, delete_ids, axis=0)

    assert len(merge_ids) == len(pick)
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    bbox_pick = ori_bbox[pick]
    #final_result = pool.map(filter_result, zip(scores_pick, merge_ids, preds_pick, pick, bbox_scores_pick))
    #final_result = [item for item in final_result if item is not None]
    # print('pPose-NMS.py:pick:',pick)

    for j in range(len(pick)):
        ids = np.arange(4)
        max_score = torch.max(scores_pick[j, ids, 0])

        # print('pPose-NMS.py:max_score1:',max_score)
        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j], ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick[j]])

        max_score = torch.max(merge_score[ids])
        # print('pPose-NMS.py:max_score2:',max_score)
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])
        # print('pPose-NMS.py:merge_pose:',merge_pose)
        # print('pPose-NMS.py:(xmax - xmin) * (ymax - ymin) < areaThres:',(xmax - xmin) * (ymax - ymin) < areaThres)

        if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < areaThres):
            continue

        final_result.append({
            'keypoints': merge_pose - 0.3,
            'kp_score': merge_score,
            'proposal_score': torch.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score),
            'box': bbox_pick[j]
        })

    return final_result







def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_preds[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    mask = (dist <= 1)

    # Define a keypoints distance
    keypoint_scores.squeeze_()
    if keypoint_scores.dim() == 1:
        keypoint_scores.unsqueeze_(0)
    if pred_scores.dim() == 1:
        pred_scores.unsqueeze_(1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = pred_scores.repeat(1, all_preds.shape[0]).transpose(0, 1)

    score_dists = torch.zeros(all_preds.shape[0], pred_scores.shape[1])
    score_dists[mask] = torch.tanh(pred_scores[mask] / delta1) * torch.tanh(keypoint_scores[mask] / delta1)

    point_dist = torch.exp((-1) * dist / delta2)
    final_dist = torch.sum(score_dists, dim=1) + mu * torch.sum(point_dist, dim=1)

    return final_dist





def PCK_match(pick_pred, all_preds, ref_dist):
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_pred[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = torch.sum(
        dist / ref_dist <= 1,
        dim=1
    )

    return num_match_keypoints


def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    '''
    dist = torch.sqrt(torch.sum(
        torch.pow(ref_pose[np.newaxis, :] - cluster_preds, 2),
        dim=2
    ))

    kp_num = 4
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = torch.zeros(kp_num, 2)
    final_score = torch.zeros(kp_num)

    if cluster_preds.dim() == 2:
        cluster_preds.unsqueeze_(0)
        cluster_scores.unsqueeze_(0)
    if mask.dim() == 1:
        mask.unsqueeze_(0)

    # Weighted Merge
    masked_scores = cluster_scores.mul(mask.float().unsqueeze(-1))
    normed_scores = masked_scores / torch.sum(masked_scores, dim=0)

    final_pose = torch.mul(cluster_preds, normed_scores.repeat(1, 1, 2)).sum(dim=0)
    final_score = torch.mul(masked_scores, normed_scores).sum(dim=0)
    return final_pose, final_score