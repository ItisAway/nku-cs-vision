# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:14:45 2020

@author: 小聪明
"""
import torch
import numpy as np
from attack_utils import FGM_Conf, L2BIM_Conf, LinfBIM_Conf, FGSM_Conf
from matplotlib import pyplot as plt
from mnist_models import get_models
from test_utils import get_test_loader, pred_from_outputs, plain_test, fixed_epsilon_attack_test
import foolbox
from utils import draw_result
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net, net_c2, net_c3 = get_models()

fgm = FGM_Conf()
l2_bim = L2BIM_Conf()
linf_bim =LinfBIM_Conf()
fgsm = FGSM_Conf()

fixed_epsilon_attack_test(net_c2, fgm, 'l2', 'FGM c2')
fixed_epsilon_attack_test(net_c2, l2_bim, 'l2', 'L2BIM c2')
fixed_epsilon_attack_test(net_c2, linf_bim, 'linf', 'LinfBIM c2')
fixed_epsilon_attack_test(net_c2, fgsm, 'linf', 'FGSM c2')


fixed_epsilon_attack_test(net_c3, fgm, 'l2', 'FGM c3')
fixed_epsilon_attack_test(net_c3, l2_bim, 'l2', 'L2BIM c3')
fixed_epsilon_attack_test(net_c3, linf_bim, 'linf', 'LinfBIM c3')
fixed_epsilon_attack_test(net_c3, fgsm, 'linf', 'FGSM c3')

from foolbox.attacks import L2FastGradientAttack
from foolbox.attacks import L2BasicIterativeAttack
from foolbox.attacks import LinfBasicIterativeAttack
from foolbox.attacks import LinfFastGradientAttack

fgm_ = L2FastGradientAttack()
l2_bim_ = L2BasicIterativeAttack()
linf_bim_ = LinfBasicIterativeAttack()
fgsm_ = LinfFastGradientAttack()
fixed_epsilon_attack_test(net, fgm_, 'l2', 'FGM LeNet5')
fixed_epsilon_attack_test(net, l2_bim_, 'l2', 'L2BIM LeNet5')
fixed_epsilon_attack_test(net, linf_bim_, 'linf', 'LinfBIM LeNet5')
fixed_epsilon_attack_test(net, fgsm_, 'linf', 'FGSM LeNet5')
'''

fgm_res = {}
fgm_res['c2'] = np.load('./res/FGM c2.npy')
fgm_res['c3'] = np.load('./res/FGM c3.npy')
fgm_res['lenet'] = np.load('./res/FGM LeNet5.npy')

l2_bim_res = {}
l2_bim_res['c2'] = np.load('./res/L2BIM c2.npy')
l2_bim_res['c3'] = np.load('./res/L2BIM c3.npy')
l2_bim_res['lenet'] = np.load('./res/L2BIM LeNet5.npy')

linf_bim_res = {}
linf_bim_res['c2'] = np.load('./res/LinfBIM c2.npy')
linf_bim_res['c3'] = np.load('./res/LinfBIM c3.npy')
linf_bim_res['lenet'] = np.load('./res/LinfBIM LeNet5.npy')

fgsm_res = {}
fgsm_res['c2'] = np.load('./res/FGSM c2.npy')
fgsm_res['c3'] = np.load('./res/FGSM c3.npy')
fgsm_res['lenet'] = np.load('./res/FGSM LeNet5.npy')

draw_result(fgm_res, 'l2', 'FGM')
draw_result(l2_bim_res, 'l2', 'L2 BIM')
draw_result(linf_bim_res, 'linf', 'Linf BIM')
draw_result(fgsm_res, 'linf', 'FGSM')





            