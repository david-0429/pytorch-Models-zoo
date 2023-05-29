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


#data loader
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    
    
#train    
def train(net, epoch):

    start = time.time()
    net.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
    
    
