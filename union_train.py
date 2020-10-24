# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:52:54 2020

@author: 小聪明
"""
from train_utils import data_expand, get_train_loader
import torch
import torch.nn.functional as F
from test_utils import get_test_loader
import os
from datetime import datetime
from binary_model import get_c2c
from union_model import UnionLeNetCn

def my_cross_entropy(x, y):
    return F.nll_loss(torch.log(x), y)

def training(net, paras):
    # GPU
    device = paras["device"]
    
    # base paragrams
    BATCH_SIZE = paras["batch_size"]
    pre_epoch = paras["pre_epoch"]
    EPOCH = paras["epoch"]
    mode1_epoch = paras["mode1_epoch"]
    
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
    
    c2c = paras["c2c"]
    expansion_times = len(c2c[0])
    
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
        if epoch <= mode1_epoch:
            net.training_mode = 1
        else:
            net.training_mode = 2
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            _, true_labels = data
            true_labels = true_labels.to(device)
            cc_images, cc_labels = data_expand(data, c2c)
            cc_images, cc_labels = cc_images.to(device), cc_labels.to(device)

            optimizer.zero_grad()
            outputs = net(cc_images)
            if net.training_mode == 1:
                loss = cost(outputs, cc_labels)
            elif net.training_mode == 3:
                loss = cost(outputs[0], cc_labels) + cost(outputs[1], true_labels)
            loss.backward()
            optimizer.step()
            
            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            if net.training_mode == 1:
                outputs = outputs[0:-1:expansion_times, :]
            if net.training_mode == 2:
                outputs = outputs[0][0:-1:expansion_times, :], outputs[1][0:-1:expansion_times, :]
            predicted = net.combine(outputs).argmax(axis=-1).to(device)
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
            if net.training_mode == 1:
                net.training_mode = 3
            if net.training_mode == 2:
                net.training_mode = 4
            correct = 0
            total = 0
            for images, labels in test_loader:
                net.eval()
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

if __name__ == "__main__":
    paras = {}
    paras["batch_size"] = 100
    paras["pre_epoch"] = 0
    paras["epoch"] = 100
    paras["mode1_epoch"] = 40
    paras["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paras["pth_folder"] = 'binary_c2_zoo'
    paras["cost"] = torch.nn.CrossEntropyLoss()
    paras["optim"] = "adam"
    paras["lr"] = 1e-3
    paras["c2c"] = get_c2c(2)
    
    #lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
    
    net = UnionLeNetCn(2)
    
    training(net, paras)