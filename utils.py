import torch
import numpy as np
import matplotlib.pyplot as plt


def pred_from_logits_train(logits, c_to_c = None):
    if c_to_c == None:
        return logits.argmax(axis=-1)
    else:
        softmax = logits.softmax(axis=-1)
        b, _  = logits.shape
        conf = torch.zeros([b, 10])
        for i in range(b):
            for j in range(10):
                conf[i, j] = softmax[i, c_to_c[j, :]].sum()
        return conf.argmax(axis=-1)

def draw_result(res_dict, lp, title):
    if lp == 'l2':
        eps_max = 4.
        thres = 1.5
    if lp == 'linf':
        eps_max = 0.5
        thres = 0.3
    # show result
    stepsize = 0.004
    if lp == 'linf':
        stepsize /= 8
    e_ = np.array(range(1,1001))*stepsize
    
    plt.figure()
    plt.xlabel(lp + 'distance')
    plt.ylabel('Accuracy')
    plt.xlim((0, eps_max))
    plt.ylim((0, 1))  
    for name, eps_res in res_dict.items():
        acc = np.zeros(1000)
        mid_ = 0
        for i in range(1000):
            acc[i] = 1 - (eps_res <= e_[i]).sum()/10000
            if mid_ == 0 and acc[i] <= 0.5:
                mid_ = e_[i]
        thres_acc = 1 - (eps_res <= thres).sum()/10000
        plt.plot(e_, acc, label='%s  %.2f/%.0f%%'%(name, mid_, thres_acc*100))
    plt.legend()
    plt.title(title)
    plt.savefig('./res/%s.svg'%title)
    plt.show()

def _imshow(img, r = 0, c = 0, i = 0, title = None):
    if r != 0:
        plt.subplot(r, c, i)
    plt.imshow(img.cpu().numpy().reshape(28,28), cmap=plt.cm.gray)
    plt.axis('off')
    if title!=None:
        plt.title(title)
        
def t2n(t):
    return t.detach().cpu().numpy()

def l2(a, b):
    return ((a - b)**2).sum().item()

def linf(a, b):
    return (a - b).max().item()

def l0(a, b):
    return (a != b).sum().item()*1.0