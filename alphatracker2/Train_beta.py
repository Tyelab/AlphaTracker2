import matplotlib.pyplot as plt
import os
import numpy as np

import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .training.yolo import test
from .training.yolo.models.experimental import attempt_load
from .training.yolo.models.yolo import Model
from .training.yolo.utils.autoanchor import check_anchors
from .training.yolo.utils.datasets import create_dataloader
from .training.yolo.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from .training.yolo.utils.google_utils import attempt_download
from .training.yolo.utils.loss import ComputeLoss
from .training.yolo.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from .training.yolo.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel

from .training.yolo import train2
from .training.yolo import opt_args

logger = logging.getLogger(__name__)



#from .training.sppe import TrainBatch
#from .training.sppe.utils import DataUtil
#from .training.sppe.utils import ImageUtil
#from .training.sppe.models import Models
#from .training.sppe import TrainBatch
#from .training.sppe import LoadModels
#from .settings import *

from .training.train_sppe import train_pose

from collections import OrderedDict

def train_object_detector(full_exp_path, model_type='yolov5s', epochs=80, batch_size=16,numWorkers=8):  # add numWorkers=0 here
    opt = opt_args.opt 
    opt.workers = numWorkers 
    opt.adam = True
    opt.project = os.path.join(full_exp_path, 'Weights', 'ObjectDetector')
    opt.data = os.path.join(full_exp_path, 'data.yaml')
    #opt.hyp = r'C:\Users\lihao\Desktop\AlphaTracker2\alphatracker2\\training\\yolo\\data\\hyp.scratch.yaml' # make general
    opt.hyp = os.path.join(os.path.dirname(__file__), os.path.join('training', 'yolo', 'data', 'hyp.scratch.yaml'))
    
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
        #check_requirements()
        
    if opt.resume:  
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori 
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)       
    
    #device = select_device(opt.device, batch_size=opt.batch_size)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    opt.device = device
    #opt.cfg = r'C:\Users\lihao\Desktop\AlphaTracker2\alphatracker2\training\yolo\models\yolov5s.yaml' # make general
    if model_type == 'yolov5s':
        opt.cfg = os.path.join(os.path.dirname(__file__), os.path.join('training', 'yolo', 'models', 'yolov5s.yaml'))
    if model_type == 'yolov5m':
        opt.cfg = os.path.join(os.path.dirname(__file__), os.path.join('training', 'yolo', 'models', 'yolov5m.yaml'))
    if model_type == 'yolov5l':
        opt.cfg = os.path.join(os.path.dirname(__file__), os.path.join('training', 'yolo', 'models', 'yolov5l.yaml'))
    if model_type == 'yolov5x':
        opt.cfg = os.path.join(os.path.dirname(__file__), os.path.join('training', 'yolo', 'models', 'yolov5x.yaml'))
    
    opt.epochs = epochs
    opt.batch_size = batch_size
    opt.total_batch_size = opt.batch_size
    
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  
        
    wandb = None
    tb_writer = None  
    if opt.global_rank in [-1, 0]:
        logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train2.train(hyp, opt, device, tb_writer, wandb)
    
    print('')
    
    
def train_pose_estimator(full_exp_path, model_type='mobilenet', epochs=300, batch_size=16, vis=1, aug=0, nThreads=6):

    weights_save_path = os.path.join(full_exp_path, 'Weights', 'PoseEstimator', 'exp')
    weights_save_path = increment_path(weights_save_path, exist_ok=False)
    os.mkdir(weights_save_path)

    #file = open(os.path.join(full_exp_path, 'setting.py'), 'r')
    #image_directory = file.readlines()[1].split('\n')[0]
    
    #file = open(os.path.join(full_exp_path, 'setting.py'), 'r')
    #json_directory = [file.readlines()[2].split('\n')[0]]
    
    #file = open(os.path.join(full_exp_path, 'setting.py'), 'r')
    #num_mouse = [int(file.readlines()[3].split('\n')[0])]
    
    #file = open(os.path.join(full_exp_path, 'setting.py'), 'r')
    #num_pose = int(file.readlines()[4].split('\n')[0])
    
    #file = open(os.path.join(full_exp_path, 'setting.py'), 'r')
    #exp_name = file.readlines()[5].split('\n')[0]
    
    #file = open(os.path.join(full_exp_path, 'setting.py'), 'r')
    #train_val_split = float(file.readlines()[6].split('\n')[0])

    
    train_pose.main(full_exp_path, weights_save_path, model_type=model_type, epochs=epochs, batch_size=batch_size, lr=1e-4, nThreads=nThreads)
    

def train_networks():
    train_object_detector()
    train_pose_estimator()
    print('')
