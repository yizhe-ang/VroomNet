import torch.nn as nn
from pretrainedmodels import dpn92


class DPN92(object):
    def __init__(self):
        self.model = dpn92(pretrained='imagenet+5k')
        
        def base_arch(pretrained=True):
            return nn.Sequential(*list(self.model.children()))
            
        self.base_arch = base_arch
    
    
    def get_model_config(self):
        return {
            'base_arch': self.base_arch,
            'cut': -1, 
            'split_on': lambda m: (m[0][0][16], m[1])
        }

    
    def get_img_stats(self):
        return [self.model.mean, self.model.std]
        
    
    def get_img_size(self):
        return self.model.input_size[1]
