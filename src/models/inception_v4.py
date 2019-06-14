import torch.nn as nn
from pretrainedmodels import inceptionv4


class InceptionV4(object):
    def __init__(self):
        self.model = inceptionv4(pretrained='imagenet')
        
        def base_arch(pretrained=True):
            all_layers = list(self.model.children())
            
            return nn.Sequential(*all_layers[0], *all_layers[1:])
            
        self.base_arch = base_arch
    
    
    def get_model_config(self):
        return {
            'base_arch': self.base_arch,
            'cut': -2, 
            'split_on': lambda m: (m[0][11], m[1])
        }

    
    def get_img_stats(self):
        return [self.model.mean, self.model.std]
        
    
    def get_img_size(self):
        return self.model.input_size[1]
