import os
import datetime
import numpy as np
import sys
import re
import argparse
import timm

import torch
import torchvision
import torchvision.transforms as transforms

            

def transformed(args, mean, std, train=True):

    if train:
      if args.DA == "non":
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])

      elif args.DA == "flip_crop":
          transform = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])

    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    
    return transform
  
  
#-------------------------------------------------------------------------------------------------------

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
