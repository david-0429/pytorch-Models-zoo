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

def extract_grad(grad_dic, model):

  for name, param in model.named_parameters():
    if param.grad is not None:
      grad_dic["layer"][name] += param.grad.mean()

  return grad_dic


def make_grad_list(num_epochs, num_batches_per_epoch, num_layers):
    grad_list = []

    for _ in range(num_epochs):
        epoch_list = []

        for _ in range(num_batches_per_epoch):
            batch_dict = {layer_num: None for layer_num in range(num_layers)}
            epoch_list.append(batch_dict)

        grad_list.append(epoch_list)

    return grad_list
#-------------------------------------------------------------------------------------------------------


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
