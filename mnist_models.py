# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:06:05 2020

@author: 小聪明
"""
# -----------------------------------------------------------------------------
# ----------------------------- DEFINE ALL MODELS -----------------------------
# ----------------------- LeNet, C_10_2, C_10_3, C_10_9 -----------------------
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import os
import itertools

# LeNet5 COMBINATION CLASS
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


# LeNet C_10_2
class LeNetC2(nn.Module):
    def __init__(self):
        super(LeNetC2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 45)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


# LeNet C_10_2 with the extra layer
class LeNetC2wel(nn.Module):
    def __init__(self, model):
        super(LeNetC2wel, self).__init__()
        self.c2 = model
        lenet = nn.Sequential(*list(model.children())[:])
        self.conv1 = lenet[0]
        self.relu1 = lenet[1]
        self.maxpool1 = lenet[2]
        self.conv2 = lenet[3]
        self.relu2 = lenet[4]
        self.maxpool2 = lenet[5]
        self.linear1 = lenet[6]
        self.relu3 = lenet[7]
        self.linear2 = lenet[8]
        self.softmax = nn.Softmax(1)
        self.antilayer = nn.Linear(45, 10)
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        # out = out.view(out.size(0), -1)
        out = self.softmax(out)
        out = self.antilayer(out)
        return out


# LeNet C_10_3
class LeNetC3(nn.Module):
    def __init__(self):
        super(LeNetC3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 120)
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


# LeNet C_10_3 with the extra layer
class LeNetC3wel(nn.Module):
    def __init__(self, model):
        super(LeNetC3wel, self).__init__()
        self.c3 = model
        lenet = nn.Sequential(*list(model.children())[:])
        self.conv1 = lenet[0]
        self.relu1 = lenet[1]
        self.maxpool1 = lenet[2]
        self.conv2 = lenet[3]
        self.relu2 = lenet[4]
        self.maxpool2 = lenet[5]
        self.linear1 = lenet[6]
        self.relu3 = lenet[7]
        self.linear2 = lenet[8]
        self.softmax = nn.Softmax(1)
        self.antilayer = nn.Linear(120, 10)
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        # out = out.view(out.size(0), -1)
        out = self.softmax(out)
        out = self.antilayer(out)
        return out


# Combination every three classes
def get_c_to_c3():
    c_10_3 = itertools.combinations(range(10), 3)
    res = []
    for i in range(10):
        res.append([])
    for i, p in zip(range(120), c_10_3):
        for ii in range(3):
            res[p[ii]].append(i)
    return res

# Combination every two classes
def get_c_to_c2():
    cc_m = torch.zeros([10,10], dtype=torch.int64)
    ii = 0
    for i in range(10):
        for j in range(i+1, 10):
            cc_m[i, j] = ii
            ii = ii + 1
    cc = torch.zeros([10,9], dtype=torch.int64)
    for i in range(10):
        cc[i, :] = torch.cat((cc_m[0:i, i], cc_m[i, i+1:10]), 0)
    return cc

def c2_antilayer_init(m):
    if isinstance(m, nn.Linear):
        if m.bias.data.shape[0] == 10:
            c_to_c = get_c_to_c2()
            m.bias.data *= 0
            m.weight.data *= 0
            for i in range(10):
                m.weight.data[i, c_to_c[i]] = 1.0/2   

def c3_antilayer_init(m):
    if isinstance(m, nn.Linear):
        if m.bias.data.shape[0] == 10:
            c_to_c = get_c_to_c3()
            m.bias.data *= 0
            m.weight.data *= 0
            for i in range(10):
                m.weight.data[i, c_to_c[i]] = 1.0/3   
                
class LeNetC9wel(nn.Module):
    def __init__(self, model):
        super(LeNetC9wel, self).__init__()
        self.c9 = model
        lenet = nn.Sequential(*list(model.children())[:])
        self.conv1 = lenet[0]
        self.relu1 = lenet[1]
        self.maxpool1 = lenet[2]
        self.conv2 = lenet[3]
        self.relu2 = lenet[4]
        self.maxpool2 = lenet[5]
        self.linear1 = lenet[6]
        self.relu3 = lenet[7]
        self.linear2 = lenet[8]
        self.softmax = nn.Softmax(1)
        self.antilayer = nn.Linear(10, 10)
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        # out = out.view(out.size(0), -1)
        out = self.softmax(out)
        out = self.antilayer(out)
        return out

def get_c_to_c9():
    res = []
    for i in range(10):
        res.append([])
    for i in range(10):
        for j in range(9):
            res[i].append((i+j)%10)
    return res

def c9_antilayer_init(m):
    if isinstance(m, nn.Linear):
        if m.weight.data.shape[0] == 10 and m.weight.data.shape[1] == 10:
            # print(m.weight.data.shape)
            # print('find')
            c_to_c = get_c_to_c9()
            m.bias.data *= 0
            m.weight.data *= 0
            for i in range(10):
                m.weight.data[i, c_to_c[i]] = 1.0/9

# return lenet lenetc2 lenetc2wel
def get_models(): 
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model LeNet5
    LeNet = LeNet5()
    LeNet.load_state_dict(torch.load('./zoo/net_120.pth'))
    LeNet.to(device)
    LeNet.eval()
    # model c_10_2
    net_c2 = LeNetC2()
    net_c2.load_state_dict(torch.load('./zoo/lenet_c_10_2.pth'))
    net_c2_wel = LeNetC2wel(net_c2)
    net_c2_wel.apply(c2_antilayer_init)
    net_c2_wel.to(device)
    net_c2_wel.eval()
    # model c_10_3
    net_c3 = LeNetC3()
    net_c3.load_state_dict(torch.load('./zoo/lenet_c_10_3.pth'))
    net_c3_wel = LeNetC3wel(net_c3)
    net_c3_wel.apply(c3_antilayer_init)
    net_c3_wel.to(device)
    net_c3_wel.eval()
    # model c_10_9
    n = LeNet5()
    n.load_state_dict(torch.load('./zoo/c9_79.pth'))
    c9 = LeNetC9wel(n)
    c9.apply(c9_antilayer_init)
    c9.to(device)
    return LeNet, net_c2_wel, net_c3_wel, c9


if __name__ == "__main__":
    from test_utils import get_test_loader, plain_test
    models = get_models()
    loader = get_test_loader(100)
    for m in models:
        plain_test(loader, m)
