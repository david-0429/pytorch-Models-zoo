import timm
import torch


def timm_resnet50(class_num, pretrain=False):
  model = timm.create_model('resnet50', pretrained=pretrain, num_classes=class_num)
  return model
