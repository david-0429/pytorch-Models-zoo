import os
import datetime
import numpy as np
import sys
import re
import argparse
import timm
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms

            

def transformed(args, mean, std, train=True):

    if train:
      if args.DA == "none":
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


def make_noisy_label(true_labels, cls_num):

    noisy_label = []
    for t_l in true_labels:
        label_list = np.arange(cls_num)

        # Delete the true label within whole label list
        label_list = np.delete(label_list, int(t_l))
        noisy_label.append(random.choice(label_list))

    noisy_labels = torch.tensor(noisy_label)
    return noisy_labels.cuda()
            
#-------------------------------------------------------------------------------------------------------

# Gradient store
def grad_store(images, targets, model):
    model.train()
    grad_dict = defaultdict(list)

    outputs = model(images)

    loss = loss_function(outputs, targets)
    loss.backward()

    # Extract gradients
    for i, (name, param) in enumerate(model.named_parameters()):
        if ('layer' in name) and ('conv' in name):
            key = name.split('.')[0]
            value = np.array(param.grad.clone().cpu())
            grad_dict[key].append(value)
    return grad_dict


# Calculate mean gradient of all batch
def calc_mean_grad(grad_batch_list):

  for i, batch_dict in enumerate(grad_batch_list):
    if i == 0:
        epoch_grad_dict = batch_dict.copy()

    else:
        for key, value in batch_dict.items():
            epoch_grad_dict[key] += value
  
  # Get mean grad vectors w.r.t. batch
    for key, value in epoch_grad_dict.items():
        epoch_grad_dict[key] = [x / len(grad_batch_list) for x in value]


  return epoch_grad_dict
            
#-------------------------------------------------------------------------------------------------------


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
