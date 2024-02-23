#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import os
import sys
sys.path.append('../..')
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_iid_v2, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, celeba_iid, celeba_iid_overlap
import logging
from GMI.utils import save_tensor_images

def get_logger(logpath):
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    formatter1 = logging.Formatter('%(filename)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter1)

    formatter2 = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')
    # os.makedirs(logpath, exist_ok=True)
    file_handler = logging.FileHandler(logpath)
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter2)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s', level=logging.DEBUG,
    #                     filename=logpath, filemode='a')

    return logger

def load_celeba_txt(train_file):
    train_set = []
    with open(train_file, 'r') as f:
        for line in f.readlines():
            train_set.append(line.strip())

    return train_set


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        # apply_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from CIFAR
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from CIFAR
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
    elif args.dataset == 'celeba':
        train_file = args.loaded_args['dataset']['train_file']
        test_file = args.loaded_args['dataset']['test_file']
        train_dataset = load_celeba_txt(train_file)
        test_dataset = load_celeba_txt(test_file)
        if args.iid:
            # user_groups = celeba_iid(train_dataset, args.num_users)
            user_groups = celeba_iid_overlap(train_dataset, args.num_users, args.samplingratio)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))  # transforms.Normalize((0.5,), (0.5,))])
            ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            # user_groups = mnist_iid_v2(train_dataset, args.num_users)
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

        # save the clients images
        if args.ToSaveUsrImg == 1:
            imgpath = os.path.join('../save/', f'UserImg/{args.timestr}')
            os.makedirs(imgpath, exist_ok=True)
            for uid in range(len(user_groups)):
                uimgpath = os.path.join(imgpath, f'client{uid+1}')
                os.makedirs(uimgpath, exist_ok=True)
                img_index = user_groups[uid]
                for imgid in img_index:
                    imgname = f'{imgid+1}_{train_dataset[imgid][1]}.png'
                    save_tensor_images(train_dataset[imgid][0], os.path.join(uimgpath, imgname))

    return train_dataset, test_dataset, user_groups


def average_weights(local_weights):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(local_weights[0])
    for key in w_avg.keys():
        for i in range(1, len(local_weights)):
            w_avg[key] += local_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(local_weights))
    return w_avg


def exp_details(args):
    args.flogger.info('\nExperimental details:')
    args.flogger.info(f'    Model     : {args.model}')
    args.flogger.info(f'    Optimizer : {args.optimizer}')
    args.flogger.info(f'    Learning  : {args.lr}')
    args.flogger.info(f'    Global Rounds   : {args.epochs}\n')

    args.flogger.info('    Federated parameters:')
    if args.iid:
        args.flogger.info('    IID')
    else:
        args.flogger.info('    Non-IID')
    args.flogger.info(f'    Fraction of users  : {args.frac}')
    args.flogger.info(f'    Local Batch size   : {args.local_bs}')
    args.flogger.info(f'    Local Epochs       : {args.local_ep}\n')
    return
