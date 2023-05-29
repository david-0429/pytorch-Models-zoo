# Pytorch Models Zoo
Image Models implements for experiments using pytorch and wandb(only Classification....)

## Requirements
- Python3
- PyTorch (> 0.4.1)
- torchvision
- numpy
- wandb

## Image Classification

### 1. Enter directory
```bash
$ cd pytorch-Models-zoo
```

### 2. Run wandb
Install wandb
```bash
$ pip install wandb
```

### 3. Training
Run ```train.py```
```
!python train.py \
-data CIFAR100 \
-name resnet \
-net resnet50 \
-epochs 30 \
-batch_size 256 \
-lr 0.01 \
-DA flip_crop \
-DA_test non \
-gpu
```
where the flags are explained as:
 - `-data`: specify the datasets of model, default: 'CIFAR100'
 - `-name`: specify the experiment name of wandb
 - `-net`: specify the classifier model network, default: 'resnet50'
 - `-epochs`: specify the number of total epochs to run, default:'200'
 - `-batch_size`: specify the mini-batch size, default: '256'
 - `-lr`: specify the initial learning rate, default: '0.01'
 - `-DA`: specify the Data Augmentation in training time, default: 'flip_crop'
 - `-DA_test`: specify the Data Augmentation in testing time, default: 'non'
 - `gpu` : use gpu or not, default: 'False'
 - 
####Models
`resnet18`
`resnet34`
`resnet50`
`resnet101`
`resnet152`
`mobilenet`
`mobilenetv2`
`shufflenet`
`shufflenetv2`
`vgg11`
`vgg13`
`vgg16`
`vgg19`
`densenet121`
`densenet161`
`densenet201`
`googlenet`
`inceptionv3`
`inceptionv4`
`inceptionresnetv2`
`xception`
`resnext50`
`resnext101`
`resnext152`
`nasnet`
`wideresnet`

### 4. Other tricks

#### Data Augmentation
 - `non` : no data augmentation
 - `flip_crop` : RandomCrop + RandomHorizontalFlip
 - `flip_crop_AA` : RandomCrop + RandomHorizontalFlip + AutoAugment
 - `flip_crop_RA` : RandomCrop + RandomHorizontalFlip + RandAugment(n=2, m=14)
