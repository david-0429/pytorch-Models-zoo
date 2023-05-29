import os
import time
import numpy as np
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from utils import transform_train, transform_test
from conf import CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD


data_path = '/content/data/CIFAR10'


def CIFAR10_train_loader(args):
  global transform_train
  transform_train = transform_train(CIFAR10_TRAIN_MEAN , CIFAR10_TRAIN_STD)

  trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
  return train_loader

def CIFAR10_test_loader(args):
  global transform_test
  transform_test = transform_test(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)

  testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
  return test_loader
