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


def transform_train():
  from train import parse
  args = parse()

  transform_train = transform_train(CIFAR10_TRAIN_MEAN , CIFAR10_TRAIN_STD, args = args.DA)
  return transform_train

def transform_test():
  from train import parse
  args = parse()

  transform_test = transform_test(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD, args = args.DA_test)
  return transform_test

def CIFAR10_train_loader():
  from train import parse
  args = parse()

  trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
  return train_loader

def CIFAR10_test_loader():
  from train import parse
  args = parse()

  testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
  return test_loader
