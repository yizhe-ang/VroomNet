import torch.nn as nn
from fastai.core import noop
from pretrainedmodels import pnasnet5large


class PNASNet5Large(object):
    def __init__(self):
        self.model = pnasnet5large(pretrained='imagenet', num_classes=1000)
        
        def identity(x): return x
        
        def base_arch(pretrained=True):    
            self.model.logits = identity
            
            return nn.Sequential(self.model)
            
        self.base_arch = base_arch
    
    
    def get_model_config(self):
        return {
            'base_arch': self.base_arch,
            'cut': noop, 
            'split_on': lambda m: (list(m[0][0].children())[8], m[1])
        }

    
    def get_img_stats(self):
        return [self.model.mean, self.model.std]
        
    
    def get_img_size(self):
        return self.model.input_size[1]
