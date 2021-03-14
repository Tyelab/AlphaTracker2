import numpy as np
import pandas as pd
import os
import shutil
import sys

from train.yolo import yolo_utils

def uniquify_results(path):

	''' credit to https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number '''
	filename, extension = os.path.splitext(path)
	counter = 1
	while os.path.exists(path):
		path = filename + "_results_version" + str(counter) + extension
		counter += 1
	return path
	
	
def uniquify_experiment(path):

	''' credit to https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number '''
	filename, extension = os.path.splitext(path)
	counter = 1
	while os.path.exists(path):
		path = filename + "_experiment_version" + str(counter) + extension
		counter += 1
	return path
	
def create_results(exp_name):
	new_path = uniquify(exp_name)
	if os.path.exists(new_path):
		#except 
		print("directory already exists, try a new name!")
	else:
		os.mkdir(new_path)
		print("made successfully")

def create_experiment(exp_name, ):
	#exp_name = 
	if os.path.exists(exp_name):
		#except 
		print("directory already exists, try a new name!")
	else:
		os.mkdir(exp_name)
		print("made successfully")
		
	
	
	
	
