#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--timestr', type=str, default='', help='')

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', '-R', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', '-K', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', '-C', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', '-E', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', '-B', type=int, default=10,
                        help="local batch size: B, and it means infinite if B=-1")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--ToSaveUsrImg', default=1, type=int, help='0 is no, 1 is yes')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--net', type=str, default='MCNN', help='network name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', type=int, default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpus', default='1,2,3,4', help='')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--lossfunc', type=str, default='crossentropy', help="specify the \
                         loss function: choices: crossentropy or NNL")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--testacc', type=float, default=0.97, help='global test accuracy')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--minrounds', type=int, default=1000, help='minmum rounds')
    parser.add_argument('--samplingratio', type=float, default=0.8, help='how many ratio of other client samples to augment client s samples')

    # attack arguments
    parser.add_argument('--attack', type=str, default='GMI', help='GMI | DMI | VMI : attack name')
    parser.add_argument('--seeds', default=200, type=int)
    parser.add_argument('--iter', default=3000, type=int)
    parser.add_argument('--save_every_epoch', default=0, type=int)

    # BiDO defense arguments
    parser.add_argument('--measure', default='COCO', help='HSIC | COCO | None')
    parser.add_argument('--lamdax', type=float, default=1, help='COCO:1, HSIC:2')
    parser.add_argument('--lamday', type=float, default=50, help='COCO:50, HSIC:20')
    parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
    # True to use BiDO; False to use MID
    parser.add_argument('--hsic_training', default=True, help='True: multi-layer constraints; False: ', type=bool)

    args = parser.parse_args()
    return args

def batch_args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                            of dataset")
    parser.add_argument('--timestr', type=str, default='', help="FL NO.")
    parser.add_argument('--targetor_name', type=str, default='', help="")
    parser.add_argument('--gpu', type=int, default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')

    # attack arguments
    parser.add_argument('--attack', type=str, default='GMI', help='GMI | DMI | VMI : attack name')
    parser.add_argument('--gpus', type=str, default='1,2', help="list of gpus")
    parser.add_argument('--rounds', type=str, default='1,2', help="list of targetor rounds")
    parser.add_argument('--gantimestr', type=str, default='', help="which gan for GMI")
    parser.add_argument('--seeds', default=200, type=int)
    parser.add_argument('--iter', default=3000, type=int)

    args = parser.parse_args()
    return args
