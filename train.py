import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from utils import train_loader
from utils import test_loader

parser = argparse.ArgumentParser(description='CIFAR-100 training')
parser.add_argument('-data', type=str, default='../data')
parser.add_argument('-name', type=str)
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('-batch_size', default=128, type=int, help='mini-batch size (default: 256)')
parser.add_argument('-lr', default=0.1, type=float, help='initial learning rate')

net = get_network(args)
args = parser.parse_args()


#wandb init
print("wandb init")
def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")
wandb.init(
    # Set the project where this run will be logged
    project="pytorch models zoo", 
    name=f"experiment_{args.name}-{get_timestamp()}"
)
    
    
#train    
def train(net, epoch):

    epoch_start_time = time.time()
    print('\epoch: %d' % epoch)
    net.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_index, (images, labels) in enumerate(train_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        
        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1), correct / total


#test
def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(test_loader):
        if arg.gpu:
            images, targets = images.cuda(), targets.cuda()
            
        outputs = net(images)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total


for epoch in range(args.epochs):

    train_loss, train_accuracy = train(net, epoch)
    test_loss, test_accuracy = test(net)
    
    wandb.log({"epoch/val_acc": test_accuracy, "epoch/val_loss": test_loss, "epoch/train_acc": train_accuracy, "epoch/trn_loss": train_loss, "epoch": epoch})

wandb.finish()
    
