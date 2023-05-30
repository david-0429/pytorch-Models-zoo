import timm
import torch


def timm_resnet50(pretrain=False, num_classes=100):
  model = timm.create_model('resnet50', pretrained=pretrain, num_classes=num_classes)
  return model
