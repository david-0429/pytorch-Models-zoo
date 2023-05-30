import timm
import torch


model = timm.create_model('resnet50', pretrained=True, num_classes=0)
