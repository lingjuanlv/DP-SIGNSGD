#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import Counter

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  300 imgs/shard X 200 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users



def mnist_noniid_class(dataset, num_users, class_per_user):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  300 imgs/shard X 200 shards
    # num_shards, num_imgs = 200, 300
    # idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    dict_class = {}

    idxs = np.arange(200*300)
    labels = dataset.train_labels.numpy()
    labels_his=Counter(labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    total_class=len(np.unique(labels))
    print('total_class',total_class)
    per_class_avg_users=int(num_users/total_class*class_per_user)
    print('per_class_avg_users',per_class_avg_users)
    remain_users=num_users*class_per_user-per_class_avg_users*total_class
    print('remain_users',remain_users)
    last_class_users=per_class_avg_users+remain_users
    print('last_class_users',last_class_users)

    idx_accumulate=0
    shard=0
    idx_shard={}
    for i in range(total_class):
        if i==total_class-1:
            per_class_avg_users=last_class_users
        idxs_per_class = idxs_labels[0, idx_accumulate:idx_accumulate+labels_his[i]]
        # print('idxs_per_class',idxs_per_class)
        num_imgs_per_user=int(len(idxs_per_class)/per_class_avg_users) 
        dict_class[i]=shard+np.arange(per_class_avg_users)
        #idx of per shard 
        for k in range(per_class_avg_users):
            idx_shard[shard+k]= idxs_per_class[k*num_imgs_per_user:(k+1)*num_imgs_per_user]

        shard+=per_class_avg_users
        print('shard',shard)
        idx_accumulate+=labels_his[i]       
    num_shard= [k for k in range(shard)]

    print(len(idx_shard)) 
    print('dict_class',dict_class)
    # divide and assign class_per_user shards/client
    # ensure each user has class_per_user
    for i in range(num_users):
        # print('user',i)
        flag=1
        while flag:
            if len(num_shard)>class_per_user:
                rand_set = np.random.choice(num_shard, class_per_user, replace=False)
                # print('rand_set',rand_set)
                dict_keys=[]
                for j in rand_set:
                    dict_keys.append([key for idx, key in enumerate(dict_class) if j in dict_class[key]])
                if len(np.unique(dict_keys))==class_per_user:
                    flag=0
                    for rand in rand_set:
                        dict_users[i] = np.concatenate((dict_users[i], idx_shard[rand]), axis=0).astype(int)
                    num_shard = list(set(num_shard) - set(rand_set))
            else:
                rand_set = num_shard
                flag=0
                for rand in rand_set:
                    dict_users[i] = np.concatenate((dict_users[i], idx_shard[rand]), axis=0).astype(int)
                num_shard = list(set(num_shard) - set(rand_set))
            print('num_shard',num_shard)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_noniid_class(dataset, num_users, class_per_user):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  300 imgs/shard X 200 shards
    # num_shards, num_imgs = 200, 300
    # idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    dict_class = {}

    idxs = np.arange(50000)
    labels = np.array(dataset.targets)
    labels_his=Counter(labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    total_class=len(np.unique(labels))
    print('total_class',total_class)
    per_class_avg_users=int(num_users/total_class*class_per_user)
    print('per_class_avg_users',per_class_avg_users)
    remain_users=num_users*class_per_user-per_class_avg_users*total_class
    print('remain_users',remain_users)
    last_class_users=per_class_avg_users+remain_users
    print('last_class_users',last_class_users)

    idx_accumulate=0
    shard=0
    idx_shard={}
    for i in range(total_class):
        if i==total_class-1:
            per_class_avg_users=last_class_users
        idxs_per_class = idxs_labels[0, idx_accumulate:idx_accumulate+labels_his[i]]
        num_imgs_per_user=int(len(idxs_per_class)/per_class_avg_users) 
        dict_class[i]=shard+np.arange(per_class_avg_users)
        #idx of per shard 
        for k in range(per_class_avg_users):
            idx_shard[shard+k]= idxs_per_class[k*num_imgs_per_user:(k+1)*num_imgs_per_user]

        shard+=per_class_avg_users
        print('shard',shard)
        idx_accumulate+=labels_his[i]    
    num_shard= [k for k in range(shard)]

    # print(idx_shard)    
    print(len(idx_shard)) 
    print('dict_class',dict_class)
    # divide and assign class_per_user shards/client
    # ensure each user has class_per_user
    for i in range(num_users):
        flag=1
        while flag:
            if len(num_shard)>class_per_user:
                rand_set = np.random.choice(num_shard, class_per_user, replace=False)
                dict_keys=[]
                for j in rand_set:
                    dict_keys.append([key for idx, key in enumerate(dict_class) if j in dict_class[key]])
                if len(np.unique(dict_keys))==class_per_user:
                    flag=0
                    for rand in rand_set:
                        dict_users[i] = np.concatenate((dict_users[i], idx_shard[rand]), axis=0).astype(int)
                    num_shard = list(set(num_shard) - set(rand_set))
            else:
                rand_set = num_shard
                flag=0
                for rand in rand_set:
                    dict_users[i] = np.concatenate((dict_users[i], idx_shard[rand]), axis=0).astype(int)
                num_shard = list(set(num_shard) - set(rand_set))
            print('num_shard',num_shard)
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)

