# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:25:04 2020

@author: Hu
"""

import torch
import torch.nn as nn
import itertools
from scipy.special import comb
def get_c2c(cn):
    c2c = itertools.combinations(range(10), cn)
    res = []
    for i in range(10):
        res.append([])
    for i, p in zip(range(int(comb(10,cn))), c2c):
        for ii in range(cn):
            res[p[ii]].append(i)
    return res

class UnionLeNetCn(nn.Module):
    def __init__(self, cn):
        super(UnionLeNetCn, self).__init__()
        self.training_mode = 0
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        if cn == 2:
            self.linear2 = nn.Linear(200, 45)
            self.c2c_layer = nn.Linear(45, 10)
        if cn == 3:
            self.linear2 = nn.Linear(200, 120)
            self.c2c_layer = nn.Linear(120, 10)
        if cn == 9:
            self.linear2 = nn.Linear(200, 10)
            self.c2c_layer = nn.Linear(10, 10)
        self.softmax = nn.Softmax(1)
        
        c2c = get_c2c(cn)
        self.c2c_layer.weight.data *= 0
        self.c2c_layer.bias.data *= 0
        for i in range(10):
            self.c2c_layer.weight.data[i, c2c[i]] = 1.0/cn   
        
        self.linear2_2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        # for training stage 1
        if self.training_mode == 1:
            out = self.linear2(out)
            return out
        # for training stage 2
        if self.training_mode == 2:
            out1 = self.linear2(out)
            out2 = self.linear2_2(out)
            return out1, out2
        # for training stage 1 testing
        if self.training_mode == 3:
            out = self.linear2(out)
            out = self.softmax(out)
            out = self.c2c_layer(out)
            return out
        # for training stage 2 testing and attack
        if self.training_mode == 4:
            out1 = self.linear2(out)
            out1 = self.softmax(out1)
            out1 = self.c2c_layer(out1)
            out2 = self.linear2_2(out)
            out2 = self.softmax(out2)
            return (out1 + out2) * 0.5
    def combine(self, x):
        if self.training_mode == 1:
            out = self.softmax(x)
            out = self.c2c_layer(out)
            return out
        if self.training_mode == 2:
            out1 = x[0]
            out1 = self.softmax(out1)
            out1 = self.c2c_layer(out1)
            out2 = x[1]
            out2 = self.softmax(out2)
            return (out1 + out2) * 0.5
        if self.training_mode == 3 or 4:
            return x
        