# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:15:22 2020

@author: 小聪明
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from test_utils import get_test_loader
import os
from datetime import datetime

train_data = torchvision.datasets.MNIST(
    './mnist', train=True, download=False, transform=torchvision.transforms.ToTensor()
)
    
# return test_loader
def get_train_loader(batch_size, train_data=train_data):
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    return train_loader

def data_expand(data, c_to_c):
    images, labels = data
    
    b, c, w, h = images.shape
    e_n = len(c_to_c[0])
    cc_images = torch.zeros([b*e_n, c, w, h], dtype=images.dtype)
    cc_labels = torch.zeros(b*e_n, dtype=labels.dtype)
    ii = 0
    for img, lbl in zip(images, labels):
        cc_images[ii*e_n:ii*e_n + e_n, :, :, :] = img.expand(e_n, c, w, h)
        cc_labels[ii*e_n:ii*e_n + e_n] = torch.tensor(c_to_c[lbl],dtype=labels.dtype)
        ii += 1
    return cc_images, cc_labels

def training(net, paras):
    # GPU
    device = paras["device"]
    
    # base paragrams
    BATCH_SIZE = paras["batch_size"]
    pre_epoch = paras["pre_epoch"]
    EPOCH = paras["epoch"]
    
    # dataset laoder
    train_loader = get_train_loader(BATCH_SIZE)
    test_loader = get_test_loader(BATCH_SIZE)
    
    # save models to pth_folder
    pth_folder = paras["pth_folder"]
    
    lr = paras["lr"]
    cost = paras["cost"]
    if paras["optim"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    if paras["optim"] == "adagrad":
        optimizer = torch.optim.Adagrad(net.parameters(), lr = lr)
    if "c2c" in paras:
        c2c = paras["c2c"]
        expansion_times = len(c2c[0])
    else:
        c2c = None
        expansion_times = 1
    
    if not os.path.exists(pth_folder):
        os.makedirs(pth_folder)
    f = open("%s/acc.txt"%pth_folder, "a")
    f2 = open("%s/log.txt"%pth_folder, "a")
    now = datetime.now()
    log_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    f.write('\n%s\n%s\n'%(('='*80), log_time))
    f2.write('\n%s\n%s\n'%(('='*80), log_time))
    
    if pre_epoch != 0:
        net.load_state_dict(
                torch.load('./%s/net_0%2d.pth'%(pth_folder, pre_epoch)))
        net.to(device)
    net.to(device)
    
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        net.on_training = True
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            _, true_labels = data
            true_labels = true_labels.to(device)
            if c2c != None:
                cc_images, cc_labels = data_expand(data, c2c)
            else:
                cc_images, cc_labels = data
            cc_images, cc_labels = cc_images.to(device), cc_labels.to(device) 
            optimizer.zero_grad()
            outputs = net(cc_images)
            loss = cost(outputs, cc_labels)
            loss.backward()
            optimizer.step()
            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            predicted = net.combine(outputs[0:-1:expansion_times, :]).argmax(axis=-1).to(device)
            correct += predicted.eq(true_labels).cpu().sum()
            total += true_labels.size(0)
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            f2.write('\n')
            f2.flush()
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            net.eval()
            net.on_training = False
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                pred = net(images).argmax(axis=-1)
                pred = pred.to(device)
                total += labels.size(0)
                correct += pred.eq(labels).cpu().sum()
            acc = 100. * correct / total
            print('测试分类准确率为：%.3f%%' % acc)
            print('Saving model......')
            torch.save(net.state_dict(), './' + pth_folder + '/net_%03d.pth' % (epoch + 1))
            f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
            f.write('\n')
            f.flush()
    f.close()
    f2.close()