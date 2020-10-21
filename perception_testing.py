from mnist_models import get_models
from test_utils import get_test_loader, pred_from_outputs
from utils import _imshow
from attack_utils import FGSM_Conf
from foolbox import PyTorchModel
import torch
_, c2, _, _ = get_models()
fc2 = PyTorchModel(c2, bounds=(0, 1))
loader = get_test_loader(20)
for b, l in loader:
    b, l = b.cuda(), l.cuda()
    break
eps = 0.0005
step_size = 0.0005
fgsm = FGSM_Conf()


b_, l_ = b, l
cln = torch.zeros_like(b)
a = torch.zeros_like(b)
i = 0
while True:
    _, adv, suc = fgsm(fc2, b_, l_, epsilons=eps)
    eps += step_size
    if suc.sum() > 0:
        a[i:i+suc.sum()] = adv[suc]
        cln[i:i+suc.sum()] = b_[suc]
        i = i + suc.sum()
        b_ = b_[~suc]
        l_ = l_[~suc]
    if i == 20:
        break
e = (cln - a).reshape([20, 28*28]).max(axis=1)



import matplotlib.pyplot as plt



p_a = pred_from_outputs(c2(a))
p_c = pred_from_outputs(c2(cln))



sfa = c2.c2(a).softmax(axis=-1)
sfc = c2.c2(cln).softmax(axis=-1)

i = 0
plt.figure(figsize=(16,64))
for a_, pa_, c_, pc_, e_, s1, s2 in zip(a, p_a, cln, p_c, e.values, sfa, sfc):
    _imshow(a_, 20, 4, i*4 + 1, str(pa_.item()) + ' %.2f'%e_.item())
    _imshow(c_, 20, 4, i*4 + 2, str(pc_.item()))
    plt.subplot(20,2,i*2 + 2)
    plt.plot(range(45), s1.detach().cpu().numpy(), label='adv')
    plt.plot(range(45), s2.detach().cpu().numpy(), label='cln')
    i += 1
    plt.legend()
plt.tight_layout()
plt.savefig('./linf_perception_conf45.svg')



from attack_utils import L2BIM_Conf
eps2 = 0.004
step_size2 = 0.004
bim2 = L2BIM_Conf()
b_2, l_2 = b, l
cln2 = torch.zeros_like(b)
a2 = torch.zeros_like(b)
i = 0
while True:
    _, adv, suc = bim2(fc2, b_2, l_2, epsilons=eps2)
    eps2 += step_size2
    if suc.sum() > 0:
        a2[i:i+suc.sum()] = adv[suc]
        cln2[i:i+suc.sum()] = b_2[suc]
        i = i + suc.sum()
        b_2 = b_2[~suc]
        l_2 = l_2[~suc]
    if i == 20:
        break

e2 = ((cln2 - a2).reshape([20, 28*28])**2).sum(axis=1)

p_a2 = pred_from_outputs(c2(a2))
p_c2 = pred_from_outputs(c2(cln2))

sfa2 = c2.c2(a2).softmax(axis=-1)
sfc2 = c2.c2(cln2).softmax(axis=-1)
i = 0
plt.figure(figsize=(16,64))
for a_, pa_, c_, pc_, e_, s3, s4 in zip(a2, p_a2, cln2, p_c2, e2, sfa2, sfc2):
    _imshow(a_, 20, 4, i*4 + 1, str(pa_.item()) + ' %.2f'%e_.item())
    _imshow(c_, 20, 4, i*4 + 2, str(pc_.item()))
    plt.subplot(20,2,i*2 + 2)
    plt.plot(range(45), s3.detach().cpu().numpy(), label='adv')
    plt.plot(range(45), s4.detach().cpu().numpy(), label='cln')
    i += 1
    plt.legend()
plt.tight_layout()
plt.savefig('./l2_perception_conf45.svg')


