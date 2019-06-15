"""Module that contains model configurations to be fed into a Trainer/Learner.
"""
import torch.nn as nn
from fastai.vision.learner import model_meta
from fastai.core import noop
from fastai.vision.models import (
    resnet50, resnet101, resnet152,
)
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


# ------------------------------------------------------------------------------
# ResNets
# ------------------------------------------------------------------------------
resnet50_config = {
    'base_arch': resnet50,
}

resnet101_config = {
    'base_arch': resnet101,
}

resnet152_config = {
    'base_arch': resnet152
}

# ------------------------------------------------------------------------------
# SEResNeXt101
# ------------------------------------------------------------------------------
def se_resnext101_model(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)

    return model

se_resnext101_config = {
    'base_arch': se_resnext101_model,
    'cut': -2,
    'split_on': lambda m: (m[0][3], m[1])
}

# ------------------------------------------------------------------------------
# PNASNet-5
# ------------------------------------------------------------------------------
def identity(x): return x

def pnasnet5large_model(pretrained=False):    
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.pnasnet5large(pretrained=pretrained, num_classes=1000) 
    model.logits = identity
    
    return nn.Sequential(model)
    
pnasnet5large_config = {
    'base_arch': pnasnet5large_model,
    'cut': noop, 
    'split_on': lambda m: (list(m[0][0].children())[8], m[1])
}

# ------------------------------------------------------------------------------
# Inception-v4 
# ------------------------------------------------------------------------------
def inceptionv4_model(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.inceptionv4(pretrained=pretrained)
    all_layers = list(model.children())
    
    return nn.Sequential(*all_layers[0], *all_layers[1:])
    
inceptionv4_config = {
    'base_arch': inceptionv4_model,
    'cut': -2, 
    'split_on': lambda m: (m[0][11], m[1])
}


# ------------------------------------------------------------------------------
# EfficientNetB3
# ------------------------------------------------------------------------------
# def efficient_net_b3_model(pretrained=True):
#     model = EfficientNet.from_pretrained('efficientnet-b3')
# 
#     return nn.Sequential(model)
# 
# efficient_net_custom_head = nn.Sequential(
#     nn.BatchNorm1d(1000),
#     nn.Dropout(0.25),
#     nn.Linear(1000, 512),
#     nn.ReLU(),
# 
#     nn.BatchNorm1d(512),
#     nn.Dropout(0.5),
#     nn.Linear(512, 196),
# )
# 
# efficient_net_b3_config = {
#     'base_arch': efficient_net_b3_model,
#     'cut': noop,
#     'split_on': lambda m: (list(m[0][0].children())[2][7], m[1]),
#     'custom_head': efficient_net_custom_head
# }
