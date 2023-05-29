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
where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
