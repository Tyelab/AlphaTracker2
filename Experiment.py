import numpy as np
import pandas as pd
import os
import shutil
import sys

from .training.yolo import yolo_utils

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
    
    os.mkdir(dirpath)
    os.mkdir(Labels_path)
    os.mkdir(Results_path)
   
   
def create_experiment(save_location, experiment_name, image_filepaths, json_filepaths, num_obj, num_parts, split):
    full_save_path = os.path.join(save_location, experiment_name)
    if os.path.exists(full_save_path):
        #update_directory()
        print('Image directory alread exists here: {}'.format(os.path.join(full_save_path, 'TrainImagesAndLabels')))
        print('If you want to re-make the directory, manually delete the current directory and call this function again!')
        print('')
        
    else:
        make_directory(full_save_path)
        save_dir = os.path.join(full_save_path, 'TrainImagesAndLabels')
        yolo_utils.make_directory(image_filepaths, json_filepaths, save_dir, num_parts, num_obj, split)
        #yolo_utils.make_yaml()
        print('Image directory has been created here: {}'.format(os.path.join(full_save_path, 'TrainImagesAndLabels')))
    
    
    return full_save_path