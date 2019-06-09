"""
Available SENet models:
- SENet154
- SE-ResNet50
- SE-ResNet101
- SE-ResNet152
- SE-ResNeXt50_32x4d
- SE-ResNeXt101_32x4d
"""
import pretrainedmodels
from fastai.vision.learner import model_meta


class SEResNeXt101(object):
    def __init__(self):
        # Wrap model for PyTorch API
        def se_resnext101(pretrained=False):
            pretrained = 'imagenet' if pretrained else None
            model = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)

            return model

        self.model = se_resnext101

        # Define splits and layer groups
        meta = { 'cut': -2,
                 'split': lambda m: (m[0][3], m[1]) }

        # Update model meta
        model_meta[se_resnext101] = meta


    def get_model(self):
        """Returns the model to be passed to a Learner.
        """
        return self.model
