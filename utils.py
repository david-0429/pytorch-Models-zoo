import os
import datetime
import numpy
import sys
import re
import argparse

import torch
import torchvision
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
args = parser.parse_args()


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

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
  
  
def transform_test(mean, std, agrs):
    if agrs.DA_test == 'non':
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
