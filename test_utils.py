import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import time
import numpy as np
import foolbox
import torch
import torchvision
from torch.utils.data import DataLoader
from utils import l2, linf, l0
from attack_utils import get_fixed_eps_attacks, get_min_attacks

# test data
test_data = torchvision.datasets.MNIST(
    './mnist', train=False, download=False, transform=torchvision.transforms.ToTensor()
)
    
# return test_loader
def get_test_loader(batch_size, test_data=test_data):
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)
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
    print('%s acc：%.3f%%' % ('plain', acc), timing(start))

def fixed_eps_attack_test(model, attack, lp, title):
    start = time.time()
    fmodel = foolbox.PyTorchModel(model, bounds=(0, 1))
    loader = get_test_loader(1)
    device = torch.device("cuda")
    eps_res = np.ones(10000)
    idx = 0
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)
        # set eps = 0 while predict incorrectly
        pred = model(img).argmax(axis=-1)
        if pred != lbl:
            eps_res[idx] = 0
            idx += 1
            continue
        # compute eps_max
        anti_bi = torch.zeros_like(img)
        anti_bi[img < 0.5] = 1
        if lp == 'l2':
            eps_max = l2(img, anti_bi)
        if lp == 'linf':
            eps_max = linf(img, anti_bi)
        # check whether attack_method can success under eps_max
        adv_max, _, suc_max = attack(fmodel, img, lbl, epsilons=eps_max)
        if suc_max == False:
            eps_res[idx] = 2.0 if lp == 'linf' else 28*28+1.0
            idx += 1
            continue
        # binary search
        e_min, e_max = 0, eps_max
        min_section = 0.00025 if lp == 'linf' else 0.002
        while (e_max - e_min) > min_section:
            eps_ = (e_min+e_max)/2
            _, _, is_ = attack(fmodel, img, lbl, epsilons=eps_)
            if is_ == True:
                e_max = eps_
            else:
                e_min = eps_
        eps_res[idx] = e_max
        idx += 1

    np.save('./res/npy/%s.npy'%title, eps_res)
    print(title + timing(start))
    return eps_res

def minim_attack_test(model, attack, lp, title, batch_size = 100):
    start = time.time()
    fmodel = foolbox.PyTorchModel(model, bounds=(0, 1))
    loader = get_test_loader(batch_size)
    eps_res = np.ones(10000)
    idx = 0
    if lp == 'l2':
        distance = l2
    elif lp == 'linf':
        distance = linf
    else:
        distance = l0
    for b, l in loader:
        b, l = b.cuda(), l.cuda()
        _, a, iss = attack(fmodel, b, l, epsilons=None)
        for b_, a_, iss_ in zip(b, a, iss):
            if iss_ == False:
                # eps_oo of l0, l2 is same
                eps_res[idx] = 2.0 if lp == 'linf' else 28*28+1.0
                idx += 1
            else:
                eps_res[idx] = distance(b_, a_)
                idx += 1

    print(title + timing(start))
    np.save('./res/npy/%s.npy'%title, eps_res)
    return eps_res
    
def adversarial_testing(model, loss, model_name, batch_size = 100):
    l2_fea, linf_fea = get_fixed_eps_attacks(loss = loss)
    l2_ma, linf_ma, l0_ma = get_min_attacks(loss = loss)
    '''
    for name, atk in l2_fea.items():
        fixed_eps_attack_test(model, atk, 'l2', name + model_name)
    for name, atk in linf_fea.items():
        fixed_eps_attack_test(model, atk, 'linf', name + model_name)
    '''
    for name, atk in l2_ma.items():
        minim_attack_test(model, atk, 'l2', name + model_name, batch_size)
    for name, atk in linf_ma.items():
        minim_attack_test(model, atk, 'linf', name + model_name, batch_size)
    for name, atk in l0_ma.items():
        minim_attack_test(model, atk, 'l0', name + model_name, batch_size)
    
    