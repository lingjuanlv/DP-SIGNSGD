#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet18
from utils import get_dataset, average_weights, exp_details, compute_grad_update,init_deterministic,aggregate_signsgd,l2norm


if __name__ == '__main__':
    init_deterministic()
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print('user_groups',user_groups)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    if args.model == 'ResNet18':
        global_model = ResNet18(args=args)        
    if args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        print('len_in',len_in)
        global_model = MLP(dim_in=len_in, dim_hidden=args.dim_hidden,
                           dim_out=args.num_classes)
    # else:
    #     exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    test_accs=[]

    lr_epoch=args.lr
    if args.server_momentum==1:
        velocity = {j: torch.zeros_like(param.detach()) for j, (pname, param) in enumerate(global_model.named_parameters())}
    for epoch in tqdm(range(args.epochs)):
        local_grads, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if args.Byzantine>0:
            # Byzantine_user=idxs_users[:int(args.Byzantine)]
            Byzantine_user=idxs_users[:int(np.floor(args.Byzantine*len(idxs_users)))] 

        print('lr_epoch:',lr_epoch)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, lr = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, lr_epoch=lr_epoch)
            print('lr:',lr)
            local_grad=compute_grad_update(args, global_model, w, lr=lr_epoch)
            if args.Byzantine>0:
                if idx in Byzantine_user:
                    local_grad=[torch.div(local_grad_i,-1) for local_grad_i in local_grad]
            local_grads.append(local_grad)  
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        if args.mode=='EF-DP-SIGNSGD':
            if epoch==0:
                residual_error=[torch.zeros(grad.shape, device=grad.device) for grad in local_grads[0]]
            global_model, residual_error = aggregate_signsgd(epoch, args, global_model,local_grads, lr=lr_epoch, residual_error=residual_error, user_groups=user_groups)
        else:
            global_model = aggregate_signsgd(epoch, args, global_model, local_grads, lr=lr_epoch, user_groups=user_groups)
        lr_epoch=lr

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_accs.append(test_acc)
        print(f' \n Results after {epoch} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        print('test_accs',test_accs)

    print('test_accs',test_accs)
    # Saving the objects train_loss and train_accuracy:
    if args.mode=='DP-SIGNSGD' or 'EF-DP-SIGNSGD':
        file_name = 'save/objects/{}_{}_class{}_E{}_C[{}]_iid[{}]_LE[{}]_B[{}]_lr[{}]_hidden[{}]_norm[{}]_momentum[{}]_eps[{}]_{}_Byzantine[{}].pkl'.\
            format(args.dataset, args.model, args.class_per_user, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs, args.lr, args.dim_hidden, args.l2_norm_clip, args.momentum, args.eps, args.mode, args.Byzantine)
    else:
        file_name = 'save/objects/{}_{}_class{}_E{}_C[{}]_iid[{}]_LE[{}]_B[{}]_lr[{}]_hidden[{}]_norm[{}]_momentum[{}]_{}_Byzantine[{}].pkl'.\
            format(args.dataset, args.model, args.class_per_user, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs, args.lr, args.dim_hidden, args.l2_norm_clip, args.momentum, args.mode, args.Byzantine)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, test_accs], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
