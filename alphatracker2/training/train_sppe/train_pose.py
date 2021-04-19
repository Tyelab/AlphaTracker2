# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

#from alphatracker2.training.train_sppe.utils.dataset import coco


import torch
import torch.utils.data
from .utils.dataset import coco
#from .utils.dataset import coco2 as coco
from .opt import opt
from tqdm import tqdm
#from .models.FastPose import createModel
from .models.FastPose22 import createModel
from .utils.eval import DataLogger, accuracy
from .utils.img import flip_v, shuffleLR_v
from .evaluation import prediction

from tensorboardX import SummaryWriter
import os
import numpy as np

from .models.sync_batchnorm import DataParallelWithCallback


if opt.sync:
    DataParallel = DataParallelWithCallback
else:
    DataParallel = torch.nn.DataParallel


def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(train_loader_desc):
        if torch.cuda.is_available():
            inps = inps.cuda().requires_grad_()
            labels = labels.cuda()
            setMask = setMask.cuda()
        else:
            inps = inps.requires_grad_()
        out = m(inps)

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar(
            'Train/Acc', accLogger.avg, opt.trainIters)

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader_desc):
        if torch.cuda.is_available():
            inps = inps.cuda()
            labels = labels.cuda()
            setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)
            
            if torch.cuda.is_available():
                flip_out = m(flip_v(inps, cuda=True))
                flip_out = flip_v(shuffleLR_v(
                    flip_out, val_loader.dataset, cuda=True), cuda=True)

                out = (flip_out.cuda() + out) / 2
                
            else:
                flip_out = m(flip_v(inps, cuda=False))
                flip_out = flip_v(shuffleLR_v(
                    flip_out, val_loader.dataset, cuda=False), cuda=False)

                out = (flip_out.cpu() + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main(full_exp_path, weights_save_path, model_type='mobilenet', epochs=300, batch_size=16, lr=1e-4, nThreads=6):
    
    
    file = open(os.path.join(full_exp_path, 'setting.py'), 'r')
    num_pose = int(file.readlines()[4].split('\n')[0])
    
    opt.annot_file_train = os.path.join(full_exp_path, 'PoseData', 'sppe', 'data', 'test1', 'data_newLabeled_01_train.h5')
    opt.annot_file_val = os.path.join(full_exp_path, 'PoseData', 'sppe', 'data', 'test1', 'data_newLabeled_01_val.h5')
    opt.img_folder_train = os.path.join(full_exp_path, 'PoseData', 'darknet', 'data', 'test1', 'color/')
    opt.img_folder_val = os.path.join(full_exp_path, 'PoseData', 'darknet', 'data', 'test1', 'color/')
    
    print(opt.annot_file_train)
    print(opt.annot_file_val)
    
    
    opt.nClasses = num_pose
    opt.nEpochs=epochs
    opt.trainBatch = batch_size
    opt.LR = lr
    opt.nThreads = nThreads

    # Model Initialize
    #m = createModel().cuda()
    #m = createModel()
    m = createModel(model_type, opt.nClasses)
    if torch.cuda.is_available():
        m.cuda()
    
    if model_type=='senet_101':
        print('using pretrained')
        #if opt.loadModel:
        #    print('Loading Model from {}'.format(opt.loadModel))
        #    current_model_weight = m.state_dict()
        #    #weight_save = torch.load(opt.loadModel)
        #    weight_save = torch.load(opt.loadModel, map_location=torch.device('cpu'))
        #    weight_save_changed = {}
        #    for k in weight_save:
        #        if 'conv_out.weight' in k or 'conv_out.bias' in k:
        #            print(k,'not used')
        #            continue
        #        weight_save_changed[k]= weight_save[k]
        #    current_model_weight.update(weight_save_changed)
        #    m.load_state_dict(current_model_weight)
   
   
   
   
   
    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():  
        criterion.cuda()

    if opt.optMethod == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=opt.LR)
    elif opt.optMethod == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=opt.LR
        )
    else:
        raise Exception

    writer = SummaryWriter(
        '.tensorboard/{}/{}'.format(opt.dataset, opt.expID))

    # Prepare Dataset
    if opt.dataset == 'coco':
        train_dataset = coco.Mscoco(train=True,img_folder=opt.img_folder_train,annot_file=opt.annot_file_train,nJoints=opt.nClasses)
        val_dataset = coco.Mscoco(train=False,img_folder=opt.img_folder_val,annot_file=opt.annot_file_val,nJoints=opt.nClasses)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.nThreads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

    # Model Transfer
    #m = DataParallel(m).cuda()
    m = DataParallel(m)
    if torch.cuda.is_available():
        m.cuda()

    fitness_train = []
    fitness_val = []
    # Start Training
    for i in range(opt.nEpochs+1):
        opt.epoch = i
        
        
        # training
        print('############# Starting Epoch {} #############'.format(opt.epoch))
        loss, acc = train(train_loader, m, criterion, optimizer, writer)
        fitness_train.append(loss)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=opt.epoch,
            loss=loss,
            acc=acc
        ))

        # save last
        opt.acc = acc
        opt.loss = loss
        m_dev = m.module
        m_dev_last = m.module
        torch.save(m_dev_last.state_dict(), os.path.join(weights_save_path, '{}.last.pkl'.format(model_type)))
        
        # validation
        loss, acc = valid(val_loader, m, criterion, optimizer, writer)
        fitness_val.append(loss)

        print('Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))
        
        # save best
        if np.argmin(fitness_val)+1 == len(fitness_val):
            m_dev_best = m.module
            torch.save(m_dev_best.state_dict(), os.path.join(weights_save_path, '{}.best.pkl'.format(model_type)))

    writer.close()


if __name__ == '__main__':
    main()
