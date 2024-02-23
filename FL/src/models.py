#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

# """

def get_in_channels(data_code):
    in_ch = -1
    if data_code == 'mnist':
        in_ch = 1
    elif data_code == 'fmnist':
        in_ch = 1
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, args.num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        # return x
        return F.log_softmax(x, dim=1)


class MCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MCNN, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2), )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(2, 2), )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        hiddens = []
        out = self.layer1(x)
        hiddens.append(out)
        out = self.layer2(out)
        hiddens.append(out)

        feature = self.layer3(out)
        hiddens.append(feature)

        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)

        return hiddens, res


class LeNet3(nn.Module):
    '''
    two convolutional layers of sizes 64 and 128, and a fully connected layer of size 1024
    suggested by 'Adversarial Robustness vs. Model Compression, or Both?'
    '''

    def __init__(self, num_classes=5, data_code='mnist'):
        super(LeNet3, self).__init__()

        in_ch = get_in_channels(data_code)

        self.conv1 = torch.nn.Conv2d(in_ch, 32, 5, 1, 2)  # in_channels, out_channels, kernel, stride, padding
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1, 2)

        # Fully connected layer
        if data_code == 'mnist':
            dim = 7 * 7 * 64
        elif data_code == 'cifar10':
            dim = 8 * 8 * 64

        self.fc1 = torch.nn.Linear(dim, 1024)  # convert matrix with 400 features to a matrix of 1024 features (columns)
        self.fc2 = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        feat = x.view(-1, np.prod(x.size()[1:]))
        x = F.relu(self.fc1(feat))
        x = self.fc2(x)

        return feat, x

class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=True):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale: 
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2)   
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),  
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x) 
        x = torch.flatten(x, 1) 
        logits = self.classifier(x) 
        probas = F.softmax(logits, dim=1)

        return probas, logits
