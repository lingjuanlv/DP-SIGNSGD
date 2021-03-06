#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations

def get_data_loaders(minibatch_size, microbatch_size, iterations, drop_last=True):
    def minibatch_loader(dataset):
        return DataLoader(
            dataset,
            batch_sampler=IIDBatchSampler(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            # Using less data than allowed will yield no worse of a privacy guarantee,
            # and sometimes processing uneven batches can cause issues during training, e.g. when
            # using BatchNorm (although BatchNorm in particular should be analyzed seperately
            # for privacy, since it's maintaining internal information about forward passes
            # over time without noise addition.)
            # Use seperate IIDBatchSampler class if a more granular training process is needed.
            drop_last=drop_last,
        )
    return minibatch_loader, microbatch_loader

def microbatch_loader(minibatch):
    return DataLoader(
        minibatch,
        batch_size=1,
        # Using less data than allowed will yield no worse of a privacy guarantee,
        # and sometimes processing uneven batches can cause issues during training, e.g. when
        # using BatchNorm (although BatchNorm in particular should be analyzed seperately
        # for privacy, since it's maintaining internal information about forward passes
        # over time without noise addition.)
        # Use seperate IIDBatchSampler class if a more granular training process is needed.
        drop_last=True,
    )

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, args, l2_norm_clip, minibatch_size, **kwargs):
            super(DPOptimizerClass, self).__init__(**kwargs)

            self.l2_norm_clip = args.l2_norm_clip
            self.minibatch_size = args.local_bs

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        # param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        # param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(**kwargs)

    return DPOptimizerClass

DPSGD = make_optimizer_class(torch.optim.SGD)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #     dataset, list(idxs))
        self.trainloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
    

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        return trainloader

    def update_weights(self, model, global_round, lr_epoch):
        # Set mode to train model
        model.train()
        epoch_loss = []
        if self.args.mode=='DP-SIGNSGD' or self.args.mode=='EF-DP-SIGNSGD':
            optimizer = DPSGD(
                args=self.args,
                l2_norm_clip=self.args.l2_norm_clip,
                minibatch_size=self.args.local_bs,
                params=model.parameters(),
                lr=lr_epoch,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.args.gamma)
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                X_minibatch, y_minibatch = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                    X_microbatch = X_microbatch.to(self.device)
                    y_microbatch = y_microbatch.to(self.device)
                    optimizer.zero_microbatch_grad()
                    log_probs = model(X_microbatch)
                    loss = self.criterion(log_probs, y_microbatch)
                    loss.backward()
                    optimizer.microbatch_step()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, 1, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))    
            scheduler.step()
            lr = scheduler.get_lr()[0]
        else:            
            # Set optimizer for the local updates
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr_epoch,
                                            momentum=self.args.momentum)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr_epoch,
                                             weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.args.gamma)
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            scheduler.step()    
            lr = scheduler.get_lr()[0]
        return model, sum(epoch_loss) / len(epoch_loss), lr

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


