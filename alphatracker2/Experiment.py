import numpy as np
import pandas as pd
import os
import shutil
import sys
import yaml

from alphatracker2.training.yolo import yolo_utils
#from .t3 import generate_pose_train
from alphatracker2.t3 import generate_pose_train

def uniquify_results(path):

	''' credit to https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number '''
	filename, extension = os.path.splitext(path)
	counter = 1
	while os.path.exists(path):
		path = filename + "_results_version" + str(counter) + extension
		counter += 1 
	return path

    
def make_directory(dirpath):
    Labels_path = os.path.join(dirpath, 'Labels')
    Results_path = os.path.join(dirpath, 'Results')
    Weights_path = os.path.join(dirpath, 'Weights')
    
    Yolo_path = os.path.join(Weights_path, 'ObjectDetector')
    Sppe_path = os.path.join(Weights_path, 'PoseEstimator')
    
    os.mkdir(dirpath)
    os.mkdir(Labels_path)
    os.mkdir(Results_path)
    os.mkdir(Weights_path)
    
    os.mkdir(Yolo_path)
    os.mkdir(Sppe_path)
   
   
def make_yaml(full_save_path):
    data = dict(
        train = os.path.join(full_save_path, 'TrainImagesAndLabels', 'images', 'train'),
        val = os.path.join(full_save_path, 'TrainImagesAndLabels', 'images', 'test'),
        nc = 1,
    )
    
    with open(os.path.join(full_save_path, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
   
   
def make_settings(full_exp_path, image_root_list, json_file_path, num_mouse, num_pose, exp_name, train_val_split):
	
    if os.path.isabs(full_exp_path) == False:
        full_exp_path = os.path.join(os.getcwd(), full_exp_path)
    
    if os.path.isabs(image_root_list[0]) == False:
        image_root_list = [os.path.join(os.getcwd(), image_root_list[0])]
        
    if os.path.isabs(json_file_path[0]) == False:
        json_file_path = [os.path.join(os.getcwd(), json_file_path[0])]
    
    
    with open(os.path.join(full_exp_path, 'setting.py'), 'w') as f:
        #f.write("import os\n")
        #f.write("gpu_id=0\n")
        f.write("{}\n".format(full_exp_path))
        f.write("{}\n".format(image_root_list[0]))
        f.write("{}\n".format(json_file_path[0]))
        f.write("{}\n".format(num_mouse))
        f.write("{}\n".format(num_pose))
        f.write("'{}'\n".format(exp_name))
        f.write("{}\n".format(train_val_split))
		
		
def create_experiment(save_location, experiment_name, image_filepaths, json_filepaths, num_obj, num_parts, split, extension):
    full_save_path = os.path.join(save_location, experiment_name)
    if os.path.exists(full_save_path):
        #update_directory()
        print('Image directory alread exists here: {}'.format(os.path.join(full_save_path, 'TrainImagesAndLabels')))
        print('If you want to re-make the directory, manually delete the current directory and call this function again!')
        print('')
        
    else:
        make_directory(full_save_path)
        save_dir = os.path.join(full_save_path, 'TrainImagesAndLabels')
        yolo_utils.make_directory(image_filepaths, json_filepaths, save_dir, num_parts, num_obj, split, extension)
        make_yaml(full_save_path)
        make_settings(full_save_path, image_filepaths, json_filepaths, num_obj, num_parts, experiment_name, split)
        generate_pose_train(full_save_path, image_filepaths, json_filepaths, [num_obj], num_parts, split, extension)
        #yolo_utils.make_yaml()
        
        print('Image directory has been created here: {}'.format(os.path.join(full_save_path, 'TrainImagesAndLabels')))
    
    
    return full_save_path
