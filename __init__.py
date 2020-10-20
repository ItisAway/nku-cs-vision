# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:24:06 2020

@author: 小聪明
"""
if __name__ == "__main__":
    import torchvision
    torchvision.datasets.MNIST(
        './mnist', train=False, download=True, transform=torchvision.transforms.ToTensor()
    )