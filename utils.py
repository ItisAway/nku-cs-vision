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
        eps_inf = 28*28*1.0
    if lp == 'linf':
        eps_max = 0.5
        thres = 0.3
        eps_inf = 1.0
    # show result
    stepsize = 0.002
    if lp == 'linf':
        stepsize /= 8
    e_ = np.array(range(1,2001))*stepsize
    
    plt.figure()
    plt.xlabel(lp + ' distance')
    plt.ylabel('Accuracy')
    plt.xlim((0, eps_max))
    plt.ylim((0, 1))
    for name, eps_res in res_dict.items():
        acc = np.zeros(2000)
        mid_ = np.median(eps_res)
        for i in range(2000):
            acc[i] = 1 - (eps_res <= e_[i]).sum()/10000
        thres_acc = 1 - (eps_res <= thres).sum()/10000
        mid_ = '%.1f'%mid_ if mid_ < eps_inf else '∞'
        plt.plot(e_, acc, label='%s  %s/%.0f%%'%(name, mid_, thres_acc*100))
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