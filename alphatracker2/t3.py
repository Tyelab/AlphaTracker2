import os
import h5py
from . import data_utils2 as data_utils
#import data_utils2 as data_utils
from imp import reload
reload(data_utils)

#from Makeh5 import generate_h5
#from Makeh5 import merge_clean_ln_split_Data



image_root_list = ['/home/npadilla/Documents/Aneesh/test_test/browtf2']
json_file_list = ['/home/npadilla/Documents/Aneesh/test_test/browtf2/ATjson.json']
num_mouse = [2]
num_pose = 4
train_val_split = 0.9
image_suffix = 'jpg'


def generate_pose_train(experiment_name, image_root_list, json_file_list, num_mouse, num_pose, train_val_split, image_suffix):

    AlphaTracker_root = os.path.join(experiment_name, 'PoseData')
    if not os.path.exists(AlphaTracker_root):
        os.makedirs(AlphaTracker_root)
        
    sppe_root = os.path.join(AlphaTracker_root, 'sppe')
    darknet_root = os.path.join(AlphaTracker_root, 'darknet')
    exp_name = 'test1'


    ln_image_dir = AlphaTracker_root + '/data/'+exp_name+'/color_image/'

    ### sppe data setting
    train_h5_file = sppe_root+ '/data/'+exp_name+'/data_newLabeled_01_train.h5'
    val_h5_file = sppe_root+ '/data/'+exp_name+'/data_newLabeled_01_val.h5'

    ### yolo data setting

    color_img_prefix = 'data/'+exp_name+'/color/'
    file_list_root = 'data/'+exp_name+'/'

    # newdir=darknet_root + '/'+ color_img_prefix
    yolo_image_annot_root =darknet_root + '/'+ color_img_prefix
    train_list_file = darknet_root+'/' + file_list_root + '/' + 'train.txt'
    val_list_file = darknet_root+'/' + file_list_root + '/' + 'valid.txt'

    valid_image_root = darknet_root+ '/data/'+exp_name+'/valid_image/'

    if not os.path.exists(sppe_root+ '/data/'):
        os.makedirs(sppe_root+ '/data/')
    if not os.path.exists(sppe_root+ '/data/'+exp_name):
        os.makedirs(sppe_root+ '/data/'+exp_name)
    if not os.path.exists(darknet_root+ '/data/'+exp_name):
        os.makedirs(darknet_root+ '/data/'+exp_name)
    if not os.path.exists(darknet_root+ '/data/'+exp_name+'/color/'):
        os.makedirs(darknet_root+ '/data/'+exp_name+'/color/')
    if not os.path.exists(AlphaTracker_root + '/data/'):
        os.makedirs(AlphaTracker_root + '/data/')
    if not os.path.exists(AlphaTracker_root + '/data/'+exp_name):
        os.makedirs(AlphaTracker_root + '/data/'+exp_name)
    if not os.path.exists(valid_image_root):
        os.makedirs(valid_image_root)
    if not os.path.exists(ln_image_dir):
        os.makedirs(ln_image_dir)

    ## evalation setting
    gt_json_file_train = AlphaTracker_root + '/data/'+exp_name+'_gt_forEval_train.json'
    gt_json_file_valid = AlphaTracker_root + '/data/'+exp_name+'_gt_forEval_valid.json'
    if not os.path.exists(AlphaTracker_root + '/data/'+exp_name):
        os.makedirs(AlphaTracker_root + '/data/'+exp_name)






    ## load and clean data
    print('*** loading and clean data from json ***')
    

    train_data,valid_data,num_allAnnot_train,num_allAnnot_valid =  data_utils.merge_clean_ln_split_Data(image_root_list,json_file_list,ln_image_dir,train_val_split,num_mouse,num_pose,valid_image_root)

    valid_len_train = len(train_data)
    valid_len_valid = len(valid_data)
    print('total training data len:',valid_len_train)
    print('total validation data len:',valid_len_valid)

    print('')


    print('generating data for training SPPE')

    data_utils.generate_h5(train_h5_file,train_data,num_allAnnot=num_allAnnot_train,num_pose=num_pose,num_mouse=num_mouse)
    data_utils.generate_h5( val_h5_file, valid_data,num_allAnnot=num_allAnnot_valid,num_pose=num_pose,num_mouse=num_mouse)

    print('training h5 file is saved as:')
    print(' ',train_h5_file)
    print('valid h5 file is saved as:')
    print(' ',val_h5_file)
    print('')



    print('generating data for training YOLO')


    data_utils.generate_yolo_data(list_file=train_list_file,\
                       data_in=train_data,\
                       image_root_in=ln_image_dir,\
                       yolo_annot_root=yolo_image_annot_root,\
                       image_suffix=image_suffix,\
                       color_img_prefix=color_img_prefix)
    data_utils.generate_yolo_data(list_file=val_list_file,\
                       data_in=valid_data,\
                       image_root_in=ln_image_dir,\
                       yolo_annot_root=yolo_image_annot_root,\
                       image_suffix=image_suffix,\
                       color_img_prefix=color_img_prefix)
                       
    for item in os.listdir(ln_image_dir):
        os.symlink(os.path.join(ln_image_dir, item), os.path.join(yolo_image_annot_root, item) )
    #os.system('ln -s {}/* {}/'.format(ln_image_dir,yolo_image_annot_root))

    print(yolo_image_annot_root)
