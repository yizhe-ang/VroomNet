import pretrainedmodels


class SEResNeXt101(object):
    def get_model(self):
        """Returns a dictionary of arguments to be passed to a Learner.

        Dictionary contains:
            - base_arch: Function that will return our base model.
            - cut (int): Where to cut out the "head" from the original model
                to add our custom model.
            - split_on: How to determine the splits of the model to form the
                layer groups (typically three groups), so that we can apply
                different learning rates during training.
        """
        def se_resnext101(pretrained=False):
            pretrained = 'imagenet' if pretrained else None
            model = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)

            return model

        return {
            'base_arch': se_resnext101,
            'cut': -2,
            'split_on': lambda m: (m[0][3], m[1])
        }
