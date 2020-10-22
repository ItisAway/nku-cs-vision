# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:31:55 2020

@author: 小聪明
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

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (torch.sign(input - 0.5) + 1)*1/2
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs
binarize = Binarize.apply
class BinaryLeNetCn(nn.Module):
    def __init__(self, cn):
        super(BinaryLeNetCn, self).__init__()
        self.on_training = True
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

    def forward(self, x):
        out = binarize(x)
        out = self.maxpool1(self.relu1(self.conv1(out)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        if self.on_training == True:
            return out
        if self.on_training == False:
            out = self.softmax(out)
            out = self.c2c_layer(out)
            return out
    
    def combine(self, x):
        out = self.softmax(x)
        out = self.c2c_layer(out)
        return out
    
    def inference(self, x):
        out = binarize(x)
        out = self.maxpool1(self.relu1(self.conv1(out)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return self.combine(out)