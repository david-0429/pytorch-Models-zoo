# models-zoo
Computer Vision models implements for experiments using pytorch

### Training
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
-DA_test non
```
