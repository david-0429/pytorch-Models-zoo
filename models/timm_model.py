import timm
import torch


def model(num_classes):
  model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
  return model
