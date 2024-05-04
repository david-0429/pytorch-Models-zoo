# Pytorch Models Zoo
Image Models implements for experiments using pytorch and wandb(only Classification....)

## Requirements
- Python3
- PyTorch (> 0.4.1)
- torchvision
- timm
- wandb

## Image Classification

### 1. Enter directory
```bash
$ cd pytorch-Models-zoo
```

### 2. Run wandb & timm
Install wandb
```bash
$ pip install wandb
```
Install timm
```bash
$ pip install timm
```

### 3. Training
Run ```train.py```
```
!python train.py \
--data CIFAR10 \
--net resnet18 \
--batch_size 256 \
--lr 0.01 \
--DA non \
--gpu
```
where the flags are explained as:
 - `--data`: specify the datasets of model, default: 'CIFAR100'
 - `--name`: specify the experiment name of wandb
 - `--net`: specify the classifier model network, default: 'resnet50'
    
 - `--epochs`: specify the number of total epochs to run, default:'200'
 - `--batch_size`: specify the mini-batch size, default: '128'
 - `--lr`: specify the initial learning rate, default: '0.001'
  
 - `--DA`: specify the Data Augmentation in training time, default: 'flip_crop'
 - `--DA_test`: specify the Data Augmentation in testing time, default: 'non'
 - `--gpu`: use gpu or not, default: 'False'

 Important!!
 - '--grad_sample_num': number of samples to store gradient, default: '1024'
 - '--noisy_comb_len': number of combinations of noisy labels, default: '10'
 - '--normal_grad_path': 
 - '--noisy_grad_path': 
 -  parser.add_argument('--grad_sample_num', default=1024, type=int, help='number of samples to store gradient')
  parser.add_argument('--noisy_comb_len', default=10, type=int, help='number of combinations of noisy labels')
  
  parser.add_argument('--normal_grad_path', default="normal_grad_dict.pt", type=str, help='file path for store normal gradient')
  parser.add_argument('--noisy_grad_path', default="noisy_grad_dict.pt", type=str, help='file path for store noisy gradient')
 


#### Data Augmentation
 - `non` : no data augmentation
 - `flip_crop` : RandomCrop + RandomHorizontalFlip
 - `flip_crop_AA` : RandomCrop + RandomHorizontalFlip + AutoAugment
 - `flip_crop_RA` : RandomCrop + RandomHorizontalFlip + RandAugment(n=2, m=14)
