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

from data.CIFAR10 import CIFAR10_train_loader, CIFAR10_test_loader
from data.CIFAR100 import CIFAR100_train_loader, CIFAR100_test_loader

from utils import get_network, adjust_learning_rate
from conf import CIFAR10_CLASS_NUM, CIFAR100_CLASS_NUM

def parse_option():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', default='CIFAR100', type=str, choices=['CIFAR10', 'CIFAR100'])
  parser.add_argument('--name', type=str)
  parser.add_argument('--net', type=str, default='resnet50', help='net type')
  parser.add_argument('--pretrain', action='store_true', default=False, help='use pretrained model or not')
  
  parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
  parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 256)')
  parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
  parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180', help='where to decay lr, can be a list')
  parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
  #parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
  
  parser.add_argument('--DA', default='flip_crop', type=str, choices=['non', 'flip_crop', 'flip_crop_AA', 'flip_crop_RA'])
  parser.add_argument('--DA_test', default='non', type=str)
  parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')

  args = parser.parse_args()

  return args

args = parse_option()


if args.data =='CIFAR10':
  if args.pretrain:
    net = get_network(args, class_num=CIFAR10_CLASS_NUM, pretrain=True)
  else:
    net = get_network(args, class_num=CIFAR10_CLASS_NUM, pretrain=False)
elif args.data =='CIFAR100':
  if args.pretrain:
    net = get_network(args, class_num=CIFAR100_CLASS_NUM, pretrain=True)
  else:
    net = get_network(args, class_num=CIFAR100_CLASS_NUM, pretrain=False)


#wandb init
print("wandb init")
def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")
  
if args.pretrain:
  wandb.init(
    # Set the project where this run will be logged
    project=f"{args.data} models zoo", 
    name=f"pretrain_{args.net}_{args.DA}_{args.batch_size}_{args.lr}-{get_timestamp()}"
)
else:
  wandb.init(
    # Set the project where this run will be logged
    project=f"{args.data} models zoo", 
    name=f"{args.net}_{args.DA}_{args.batch_size}_{args.lr}-{get_timestamp()}"
)


#Data_loader
if args.data == 'CIFAR10':
  train_loader = CIFAR10_train_loader(args)
  test_loader = CIFAR10_test_loader(args)
elif args.data == 'CIFAR100':
  train_loader = CIFAR100_train_loader(args)
  test_loader = CIFAR100_test_loader(args)


iterations = args.lr_decay_epochs.split(',')
args.lr_decay_epochs = list([])
for it in iterations:
  args.lr_decay_epochs.append(int(it))
  
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

#other tricks
'''
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
iter_per_epoch = len(train_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
'''
    
#train    
def train(net, epoch):

    epoch_start_time = time.time()
    print('epoch: %d' % epoch)
    net.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_index, (images, labels) in enumerate(train_loader):

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().float().item()
        
        b_idx = batch_index
        
        '''
        other tricks : warmup
        
        if epoch <= args.warm:
            warmup_scheduler.step()
        '''

        wandb.log({"epoch/train_acc": correct / total * 100, "epoch/trn_loss": train_loss / (b_idx + 1), "epoch": epoch})


    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1), correct / total * 100


#test
def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(test_loader):
        
        images, targets = images.cuda(), targets.cuda()
            
        outputs = net(images)
        loss = loss_function(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        
        b_idx = batch_idx

        wandb.log({"epoch/val_acc": correct / total * 100, "epoch/val_loss": test_loss / (b_idx + 1), "epoch": epoch})


    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total * 100


for epoch in range(args.epochs):

    adjust_learning_rate(epoch, args, optimizer)
    
    '''
    other tricks
    
    if epoch > args.warm:
       train_scheduler.step(epoch)
    '''
    
    train_loss, train_accuracy = train(net, epoch)
    test_loss, test_accuracy = test(net)
    
    print("-------------------------------------------------------------------------")
    
    #wandb.log({"epoch/val_acc": test_accuracy, "epoch/val_loss": test_loss, "epoch/train_acc": train_accuracy, "epoch/trn_loss": train_loss, "epoch": epoch})

wandb.finish()
    
