from pretrainedmodels import se_resnext101_32x4d


class SEResNeXt101(object):
    def __init__(self):
        self.model = se_resnext101_32x4d(pretrained='imagenet')
        
        def base_arch(pretrained=True):
            return self.model
            
        self.base_arch = base_arch
    
    
    def get_model_config(self):
        """Returns a dictionary of arguments to be passed to a Learner.

        Dictionary contains:
            - base_arch: Function that will return our base model.
            - cut (int): Where to cut out the "head" from the original model
                to add our custom model.
            - split_on: How to determine the splits of the model to form the
                layer groups (typically three groups), so that we can apply
                different learning rates during training.
        """
        return {
            'base_arch': self.base_arch,
            'cut': -2,
            'split_on': lambda m: (m[0][3], m[1])
        }

    
    def get_img_stats(self):
        return [self.model.mean, self.model.std]
        
    
    def get_img_size(self):
        return self.model.input_size[1]
