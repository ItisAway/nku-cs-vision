import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import time
import numpy as np
import foolbox
import torch
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
# test data
test_data = torchvision.datasets.MNIST(
    './mnist', train=False, download=False, transform=torchvision.transforms.ToTensor()
)
    
# return test_loader
def get_test_loader(batch_size, test_data=test_data):
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)
    return test_loader

def timing(start):
    end = time.time()
    return 'time: %.3fs'%(end - start) 

def pred_from_outputs(outputs):
    return outputs.argmax(axis=-1)

def plain_test(test_loader, net):
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = pred_from_outputs(net(images)).to(device)
        total += labels.size(0)
        correct += pred.eq(labels).cpu().sum()
    acc = 100. * correct / total
    print('%20s acc：%.3f%%' % ('plain', acc), timing(start))

def fixed_epsilon_attack_test(model, attack, lp, title):
    start = time.time()
    fmodel = foolbox.PyTorchModel(model, bounds=(0, 1))
    if lp == 'l2':
        eps_min = 0.
        eps_max = 4.
        thres = 1.5
    if lp == 'linf':
        eps_min = 0.
        eps_max = 0.5
        thres = 0.3
    loader = get_test_loader(1)
    device = torch.device("cuda")
    eps_res = np.ones(10000)
    idx = 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        e_min, e_max = eps_min, eps_max
        adv_min, _, suc_min = attack(fmodel, img, lbl, epsilons=e_min)
        adv_max, _, suc_max = attack(fmodel, img, lbl, epsilons=e_max)
        if suc_max == False:
            eps_res[idx] = eps_max + 0.5
            idx += 1
            continue
        else:
            bs_i = 0
            while(bs_i <= 10):
                eps_ = (e_min+e_max)/2
                _, _, is_ = attack(fmodel, img, lbl, epsilons=eps_)
                if is_ == True:
                    e_max = eps_
                else:
                    e_min = eps_
                bs_i += 1
            eps_res[idx] = e_max
            idx += 1
    # show result
    stepsize = 0.004
    if lp == 'linf':
        stepsize /= 8
    e_ = np.array(range(1,1001))*stepsize
    acc = np.zeros(1000)
    mid_ = 0
    for i in range(1000):
        acc[i] = 1 - (eps_res <= e_[i]).sum()/10000
        if mid_ == 0 and acc[i] <= 0.5:
            mid_ = e_[i]
    plt.figure()
    plt.plot(e_, acc)
    plt.xlabel(lp + 'distance')
    plt.ylabel('Accuracy')
    thres_acc = 1 - (eps_res <= thres).sum()/10000
    plt.title('%s    %.2f/%.0f%%    %s'%(title, mid_, thres_acc*100, timing(start)))
    plt.xlim((0, eps_max))
    plt.ylim((0, 1))        
    plt.savefig('./res/%s.svg'%title)
    plt.show()
    np.save('./res/%s.npy'%title, eps_res)
    return eps_res