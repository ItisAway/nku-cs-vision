# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:14:45 2020

@author: 小聪明
"""
'''
# -----------------------------------------------------------------------------
#                        C2 Testing & Binary C2 Testing
# -----------------------------------------------------------------------------
from mnist_models import get_models
from test_utils import adversarial_testing
import os
res_folder = './res/npy'
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
_, c2, _, _ = get_models()
adversarial_testing(c2, 'standard_ce', model_name = '__c2')
'''


# -----------------------------------------------------------------------------
#                           Draw result from .npy
# -----------------------------------------------------------------------------
import numpy as np
from utils import draw_result
gaussian_noise = {}
gaussian_noise['C2'] = np.load('./res/npy/Gaussian Noise__c2.npy')
gaussian_noise['Binary C2'] = np.load('./res/npy/Gaussian Noise__c2.npy')
#gaussian_noise['C9'] = np.load('./res/npy/Gaussian Noise__c2.npy')
#gaussian_noise['Binary C9'] = np.load('./res/npy/Gaussian Noise__c2.npy')
#gaussian_noise['LeNet5'] = np.load('./res/npy/Gaussian Noise__c2.npy')
#gaussian_noise['Binary LeNet5'] = np.load('./res/npy/Gaussian Noise__c2.npy')
draw_result(gaussian_noise, 'l2', 'Gaussian Noise')









            