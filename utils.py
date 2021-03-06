#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import random
import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal,mnist_noniid_class,cifar_noniid_class
from sampling import cifar_iid, cifar_noniid
from update import LocalUpdate, test_inference

from math import exp, sqrt
from scipy.special import erf

def init_deterministic():
    # call init_deterministic() in each run_experiments function call

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1234)

def Phi(t):
    return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS=1, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)
    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma

def sign(grad):
    return [torch.sign(update) for update in grad]

def dpvalue(p):
  return torch.tensor(1) if np.random.random() < p else torch.tensor(-1)

def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
        flattened = flattened[n_params:]

    return grad_update

def dpsign(args, grad):
    sigma=calibrateAnalyticGaussianMechanism(epsilon=args.eps, delta=args.delta, GS=args.l2_norm_clip)
    result=[]
    for update in grad:
        result.append(torch.tensor([dpvalue(Phi(i/sigma)) for i in flatten(update)]).reshape(update.shape))
    return result


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
            else:
                user_groups = cifar_noniid_class(train_dataset, args.num_users, args.class_per_user)
                    
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = 'data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid_class(train_dataset, args.num_users, args.class_per_user)

    return train_dataset, test_dataset, user_groups

def momentum(model, velocity, grad, lr):
    gamma = .9
    layer_no=0
    for param_model, param_update in zip(model.parameters(), grad):
        velocity[layer_no] = gamma * velocity[layer_no] + lr * param_update.data
        param_model.data -= velocity[layer_no]
        layer_no+=1
    return model,velocity

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def l2norm(grad):
    return torch.sum(torch.pow(flatten(grad), 2))

def compute_grad_update(args, old_model, new_model, lr, device=None):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data)/(-lr) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
    assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"
    
    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight

def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]
            
    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model

def aggregate_signsgd(epoch, args, global_model, grad_updates, lr, device=None, residual_error=None, test_dataset=None, user_groups=None):
    if grad_updates:
        len_first = len(grad_updates[0])
        assert all(len(i) == len_first for i in grad_updates), "Different shapes of parameters. Cannot aggregate."
    else:
        return

    grad_updates_ = [copy.deepcopy(grad_update) for i, grad_update in enumerate(grad_updates)]
    aggregated_gradient_updates=[]

    if device:
        for i, grad_update in enumerate(grad_updates_):
            grad_updates_[i] = [param.to(device) for param in grad_update]

    server_update = [torch.zeros(grad.shape, device=grad.device) for grad in grad_updates_[0]]
    if args.mode=='FedAvg':
        all_records=0
        for i in range(args.num_users):
            all_records+=len(user_groups[i])
        for i in range(len(grad_updates)):
            if args.weighted:
                add_gradient_updates(server_update, grad_updates[i], weight = len(user_groups[i])/all_records)
            else:
                add_gradient_updates(server_update, grad_updates[i])
        aggregated_sgd=server_update
    if args.mode=='SIGNSGD':
        for i in range(len(grad_updates)):
            add_gradient_updates(server_update, sign(grad_updates[i]))
        aggregated_signsgd=sign(server_update)
    if args.mode=='DP-SIGNSGD':
        print(args.mode)
        for i in range(len(grad_updates)):
            add_gradient_updates(server_update, dpsign(args, grad_updates[i]))
        aggregated_signsgd=sign(server_update)
    if args.mode=='EF-DP-SIGNSGD':
        print(args.mode)
        for i in range(len(grad_updates)):
            add_gradient_updates(server_update, dpsign(args, grad_updates[i]))

        aggregated_signsgd=sign([torch.div(server_update[k],args.num_users)+residual_error[k] for k in range(len(server_update))])
        residual_error=[(1-args.error_decay)*(torch.div(server_update[k],args.num_users)-torch.div(aggregated_signsgd[k],args.num_users))+args.error_decay*residual_error[k] for k in range(len(server_update))]
        

    if args.mode=='FedAvg':
        add_update_to_model(global_model, aggregated_sgd, weight=-1.0 * lr)
    else:
        add_update_to_model(global_model, aggregated_signsgd, weight=-1.0 * lr)

    if args.mode=='EF-DP-SIGNSGD':
        return global_model, residual_error
    else:
        return global_model

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
