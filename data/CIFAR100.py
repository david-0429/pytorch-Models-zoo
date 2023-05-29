import os
import time
import numpy as np
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from utils import transform_train, transform_test
from conf import CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD
from train import parse

data_path = '/content/data/CIFAR100'
args = parse()

transform_train = transform_train(CIFAR100_TRAIN_MEAN , CIFAR100_TRAIN_STD, args.DA)
transform_test = transform_test(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, args.DA_test)

def CIFAR100_train_loader():
  trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
  
  return train_loader

def CIFAR100_test_loader():
  testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
  
  return test_loader
