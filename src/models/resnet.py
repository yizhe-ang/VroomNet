from fastai.vision.models import (
    resnet50, resnet101, resnet152,
)
from fastai.vision import imagenet_stats

img_size = 224


class ResNet50(object):
    def __init__(self):
        self.base_arch = resnet50
    
    
    def get_model_config(self):
        return {
            'base_arch': self.base_arch,
        }

    
    def get_img_stats(self):
        return imagenet_stats
        
        
    def get_img_size(self):
        return img_size
        
        
class ResNet101(object):
    def __init__(self):
        self.base_arch = resnet101
    
    
    def get_model_config(self):
        return {
            'base_arch': self.base_arch,
        }

    
    def get_img_stats(self):
        return imagenet_stats
        
        
    def get_img_size(self):
        return img_size
        
        
class ResNet152(object):
    def __init__(self):
        self.base_arch = resnet152
    
    
    def get_model_config(self):
        return {
            'base_arch': self.base_arch,
        }

    
    def get_img_stats(self):
        return imagenet_stats
        
        
    def get_img_size(self):
        return img_size
