import json
import numpy as np
import pandas as pd
import os
import cv2
from tqdm.notebook import tqdm
import shutil
import random


def convert(image_filepaths, json_filepaths, save_path, extension):
    #save_path = '/content/drive/My Drive/TRAINING_DATA'
    #print(save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
        print("Creating object detector image directory: {}".format(save_path))
    else:
        os.mkdir(save_path)
        print("Creating object detector image directory: {}".format(save_path))
        #print("Didn't exist, but made anyways")
    
    comb_pop = []
    for j in range(0, len(image_filepaths)):
        #print(json_filepaths)
    #j = 0
        with open(json_filepaths[j]) as f:
            data = json.load(f)
    
        pop = []
        single_img_data_count = 0
        for i in data:
            img_name = i['filename']
            full_path = os.path.join(image_filepaths[j], img_name)
            new_name = '%04d'%(j)+'_'+'%06d'%(single_img_data_count)+'.'+extension
    
            i['filename'] = new_name
            
            if os.path.isfile(full_path):
                shutil.copy(full_path, save_path)
                
            reloc_path = os.path.join(save_path, img_name)
            os.rename(reloc_path, os.path.join(save_path, new_name))
            #os.rename(full_path, os.path.join(image_filepaths[j], new_name))
    
            single_img_data_count += 1
            pop.append(i)
            
        comb_pop.append(pop)
        
    comb_pop = [x for y in comb_pop for x in y]
    
    json_save = save_path + '/ATjson.json'
    with open(json_save, 'w') as outfile:
        json.dump(comb_pop, outfile, indent=4)
        
        
def make_directory(image_dir, json_dir, new_image_dir, num_poses, num_animals, train_test_split):
    convert(image_dir, json_dir, new_image_dir, 'jpg')
    
    with open(os.path.join(new_image_dir, 'ATjson.json'), 'r') as f:
        data = json.load(f)
        
        
    out_all = []
    data = tqdm(data)
    for j, i in enumerate(data):
        filename = i['filename']
        img = cv2.imread(os.path.join(new_image_dir, filename))
        img_y = img.shape[0]
        img_x = img.shape[1]
	    
        idd = i['annotations']
        where_face = np.where(np.array([i['class'] for i in idd]) == 'Face')[0]
        if len(where_face) != num_animals:
            print("image {} has faulty labels".format(filename))
            continue
	    
        out_ = []
        for where in where_face:
            out_.append([int(0), (idd[where]['x']+idd[where]['width']/2)/img.shape[1], 
			  (idd[where]['y']+idd[where]['height']/2)/img.shape[0], 
			  idd[where]['width']/img.shape[1], 
			  idd[where]['height']/img.shape[0]])		
        np.savetxt(os.path.join(new_image_dir, os.path.splitext(filename)[0]+".txt"), np.array(out_), fmt='%f')
        
       
        
    within_images_dir = os.path.join(new_image_dir, 'images')
    within_labels_dir = os.path.join(new_image_dir, 'labels')

    if os.path.isdir(within_images_dir) != True:
        os.mkdir(within_images_dir)
    
    if os.path.isdir(within_labels_dir) != True:
        os.mkdir(within_labels_dir)
    

    train_img = os.path.join(within_images_dir, 'train') 
    test_img = os.path.join(within_images_dir, 'test') 
    train_label = os.path.join(within_labels_dir, 'train')
    test_label = os.path.join(within_labels_dir, 'test')

    if os.path.isdir(train_img) != True:
        os.mkdir(train_img)
    
    if os.path.isdir(test_img) != True:
        os.mkdir(test_img)
    
    if os.path.isdir(train_label) != True:
        os.mkdir(train_label)
    
    if os.path.isdir(test_label) != True:
        os.mkdir(test_label)
        
        
    txt_files = []
    file_rec = []
    for file in os.listdir(new_image_dir):
        if file.endswith(".txt"):
            file_rec.append(file)
            txt_files.append(os.path.join(new_image_dir, file))
   
    random.shuffle(file_rec)
    img_rec = [os.path.splitext(i)[0]+'.jpg' for i in file_rec]

    #file_rec_ = [os.path.join(new_image_dir, f) for f in file_rec]
    #img_rec_ = [os.path.join(new_image_dir, f) for f in img_rec]

    train_images_ = file_rec[0: int(len(txt_files)*train_test_split)]
    test_images_ = file_rec[int(len(txt_files)*train_test_split): ]

    train_labels_ = img_rec[0: int(len(txt_files)*train_test_split)]
    test_labels_ = img_rec[int(len(txt_files)*train_test_split): ]
    
    
    for i, j in zip(train_images_, train_labels_):
        shutil.move(os.path.join(new_image_dir, i), os.path.join(train_label, i))
        shutil.move(os.path.join(new_image_dir, j), os.path.join(train_img, j))
    
    
    for k, m in zip(test_images_, test_labels_):
        shutil.move(os.path.join(new_image_dir, k), os.path.join(test_label, k))
        shutil.move(os.path.join(new_image_dir, m), os.path.join(test_img, m))
	    
        
        
        
#def make_yaml():

    

