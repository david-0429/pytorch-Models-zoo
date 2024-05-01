import torch
import torchvision

from utils import transformed
from conf import CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD


data_path = '/content/data/CIFAR100'


def CIFAR100_loader(args, is_train=True):
  global transform
  transform = transformed(args, CIFAR100_TRAIN_MEAN , CIFAR100_TRAIN_STD, train=is_train)
  
  if is_train:
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
  
  else:
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
  
  return data_loader
