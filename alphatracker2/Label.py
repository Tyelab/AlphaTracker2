import os
import numpy as np
import json
import cv2


def get_frames(video_directory, method='sequential', num_frames_total=200, num_frames_per_vid=[], save_location=''):
    videos = os.listdir(video_directory)
    
    # check data
    if num_frames_per_vid:
        assert len(num_frames_per_vid) == len(videos), \
        "length of 'num_frames_per_vid' must match the number of videos in 'video_directory'"

        num_frames_per = num_frames_per_vid
        num_frames_total = np.sum(num_frames_per)
        print('total frames to be extracted: {}'.format(num_frames_total))
        print('frames per video: {}'.format(num_frames_per))
    else:
        num_frames_per = np.repeat(int(num_frames_total/len(videos)), len(videos)).tolist()
        print('total frames to be extracted: {}'.format(num_frames_total))
        print('frames per video: {}'.format(num_frames_per))
        
       
    # make save directory
    where_save = save_location
    if os.path.exists(where_save):
        raise Exception("{} already exists! Try another name".format(where_save))
        #raise
    else:
        os.mkdir(where_save)
        
        
    # main loop
    for vid, num_frames in zip(videos, num_frames_per):
        video = cv2.VideoCapture( os.path.join(video_directory, vid) )
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT); 

        ranges = np.arange(0, int(total_frames))
        floor = int(np.floor(len(ranges)/num_frames))
        ranges = ranges[0::floor][0:-1]

        for frame_count in range(0,int(total_frames)):
            ret, read_frame = video.read()
            if ret == False:
                break
            if frame_count in ranges:
                cv2.imwrite(os.path.join( where_save, vid.split('.')[0]+'_frame_{}.jpg'.format(frame_count) ), 
                            read_frame)

    
    json_label_path = os.path.join(where_save, 'alphatracker_labels.json')
    total_image_list = os.listdir(where_save)
    fs = open(json_label_path, 'w')
    annotations= []
    for im in total_image_list:
        annotations.append({'annotations': [], 'class': 'image', 'filename': im})
        
    json.dump(annotations, fs, indent=4, separators=(',', ': '), sort_keys=True)
    fs.write('\n')
    fs.close()
    
    print("finished, images located at: {}".format(where_save))
