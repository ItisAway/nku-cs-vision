# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:06:05 2020

@author: 小聪明
"""
# -----------------------------------------------------------------------------
# ----------------------------- DEFINE ALL MODELS -----------------------------
# -------------------- pure / binary LeNet, C_10_2, C_10_9 --------------------
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import os
import itertools
from scipy.special import comb


def get_c2c(cn):
    c2c = itertools.combinations(range(10), cn)
    res = []
    for i in range(10):
        res.append([])
    for i, p in zip(range(int(comb(10, cn))), c2c):
        for ii in range(cn):
            res[p[ii]].append(i)
    return res


def get_c_to_c9():
    res = []
    for i in range(10):
        res.append([])
    for i in range(10):
        for j in range(9):
            res[i].append((i+j) % 10)
    return res


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (torch.sign(input - 0.5) + 1)*1/2

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


binarize = Binarize.apply


class LeNetCn(nn.Module):
    def __init__(self, cn, is_binary=False):
        super(LeNetCn, self).__init__()
        self.cn = cn
        self.is_binary = is_binary
        self.on_training = False
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, int(comb(10, self.cn)))
        if self.cn != 1:
            self.softmax = nn.Softmax(1)
            self.c2c_layer = nn.Linear(int(comb(10, self.cn)), 10)
            c2c = get_c2c(self.cn) if self.cn != 9 else get_c_to_c9()
            self.c2c_layer.weight.data *= 0
            self.c2c_layer.bias.data *= 0
            for i in range(10):
                self.c2c_layer.weight.data[i, c2c[i]] = 1.0/self.cn

    def forward(self, x):
        if self.is_binary == True:
            x = binarize(x)
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        if self.cn != 1 and self.on_training == False:
            out = self.softmax(out)
            out = self.c2c_layer(out)
        return out

    def combine(self, x):
        if self.cn == 1:
            return x
        out = self.softmax(x)
        out = self.c2c_layer(out)
        return out


def get_model(model_num, is_binary=False):
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model LeNet5
    net = LeNetCn(model_num, is_binary=is_binary)
    if model_num == 1 and not is_binary:
        net.load_state_dict(torch.load('./zoo/lenet.pth'))
    elif model_num == 1 and is_binary:
        net.load_state_dict(torch.load('./zoo/bin_lenet_070.pth'))
    elif model_num == 2 and not is_binary:
        net.load_state_dict(torch.load('./zoo/new_lenet_c_10_2.pth'))
    elif model_num == 2 and is_binary:
        net.load_state_dict(torch.load('./zoo/bic2.pth'))
    elif model_num == 9 and not is_binary:
        net.load_state_dict(torch.load('./zoo/new_c9_79.pth'))
    elif model_num == 9 and is_binary:
        net.load_state_dict(torch.load('./zoo/bin_c_10_9_net_067.pth'))
    net.to(device)
    net.eval()
    net.on_training = False
    return net


if __name__ == "__main__":
    from test_utils import get_test_loader, plain_test
    loader = get_test_loader(100)
    cn = [1,2,9]
    is_b = [True, False]
    for cn_ in cn:
        for is_b_ in is_b:
            print(cn_, 'is binary:', is_b_)
            net = get_model(cn_, is_binary=is_b_)
            plain_test(loader, net)
            print("-" * 50)

    
    
