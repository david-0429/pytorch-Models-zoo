import os
import sys
import argparse
import time
from datetime import datetime
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from data.CIFAR10 import CIFAR10_loader
from data.CIFAR100 import CIFAR100_loader

#from model import get_network
from conf import CIFAR10_CLASS_NUM, CIFAR100_CLASS_NUM

import timm


def get_network(args, class_num=10, pretrain=False):
    """ return given network
    """

    model = timm.create_model(args.model, pretrained=pretrain, num_classes=class_num)

    if args.gpu: #use_gpu
        model = model.cuda()

    return model


def parse_option():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'])
  parser.add_argument('--name', type=str)
  parser.add_argument('--model', type=str, default='resnet18', help='net type')
  parser.add_argument('--pretrain', default=False, help='use pretrained model or not')
  
  parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
  parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
  parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
  '''
  parser.add_argument('--lr_decay', default=False, type=bool, help='learning rate decay')
  parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180', help='where to decay lr, can be a list')
  parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
  '''
  parser.add_argument('--DA', default='flip_crop', type=str, choices=['non', 'flip_crop', 'flip_crop_AA', 'flip_crop_RA'])
  parser.add_argument('--DA_test', default='non', type=str)
  parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')

  args = parser.parse_args()

  return args

args = parse_option()


#wandb init
print("wandb init")
def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")

wandb.init(
  # Set the project where this run will be logged
  project=f"AAAI 2024 coming soon", 
  name=f"{args.data}_{args.model}_{args.batch_size}_{args.lr}_{get_timestamp()}"
)
wandb.config.update(args)

#Data_loader
if args.data == 'CIFAR10':
  train_loader = CIFAR10_loader(args, is_train=True)
  test_loader = CIFAR10_loader(args, is_train=False)
elif args.data == 'CIFAR100':
  train_loader = CIFAR100_loader(args, is_train=True)
  test_loader = CIFAR100_loader(args, is_train=False)
  
  
#model 
if args.data =='CIFAR10':
  model = get_network(args, class_num=CIFAR10_CLASS_NUM, pretrain=args.pretrain)
elif args.data =='CIFAR100':
  model = get_network(args, class_num=CIFAR100_CLASS_NUM, pretrain=args.pretrain)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr) # momentum=0.9, weight_decay=5e-4
    

#train    
def train(model, epoch):

    epoch_start_time = time.time()
    print('epoch: %d' % epoch)
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_index, (images, labels) in enumerate(train_loader):

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().float().item()
        
        b_idx = batch_index

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('train_Loss: %.3f | train_Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    wandb.log({"epoch/train_acc": correct / total * 100, "epoch/trn_loss": train_loss / (b_idx + 1), "epoch": epoch})

    return train_loss / (b_idx + 1), correct / total * 100


#val
def validation(model):
    epoch_start_time = time.time()
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(test_loader):
        
        images, targets = images.cuda(), targets.cuda()
            
        outputs = model(images)
        loss = loss_function(outputs, targets)

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        
        b_idx = batch_idx

    print('Validation \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('validation_Loss: %.3f | validation_Acc: %.3f%% (%d/%d)' % (val_loss / (b_idx + 1), 100. * correct / total, correct, total))
    
    wandb.log({"epoch/val_acc": correct / total * 100, "epoch/val_loss": val_loss / (b_idx + 1), "epoch": epoch})

    return val_loss / (b_idx + 1), correct / total * 100


for epoch in range(args.epochs):
    
    train_loss, train_accuracy = train(model, epoch)
    test_loss, test_accuracy = validation(model)
    
    print("-------------------------------------------------------------------------")
    
wandb.finish()
    
