import os
import datetime
import numpy
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import models

args = parser.parse_args()


def get_network():

def transform_train(mean, std, agrs):
    if args.DA == "non":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif args.DA == "flip_crop":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif args.DA == "flip_crop_AA":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

#RandAugment : N = {1, 2} and M = {2, 6, 10, 14}  
#best {N, M} in WideResNet-28-2 and Wide-ResNet-28-10 : {1,2}, {2, 14}     Can more strong M??       
    elif args.DA == "flip_crop_RA":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 14),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
      
    return train_transform
  
  
def transform_test(mean, std):
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
    ])
    
    return transform_test
  
#-------------------------------------------------------------------------------------------------------#

def CIFAR_mean_std(cifar_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar_dataset[i][1][:, :, 0] for i in range(len(cifar_dataset))])
    data_g = numpy.dstack([cifar_dataset[i][1][:, :, 1] for i in range(len(cifar_dataset))])
    data_b = numpy.dstack([cifar_dataset[i][1][:, :, 2] for i in range(len(cifar_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std
