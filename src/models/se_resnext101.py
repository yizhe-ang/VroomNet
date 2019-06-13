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


class SEResNeXt101(object):
    def __init__(self):
        # Wrap model for PyTorch API
        def se_resnext101(pretrained=False):
            pretrained = 'imagenet' if pretrained else None
            model = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)

            return model

        self.model = se_resnext101

        # Define splits and layer groups
        meta = {
            'cut': -2,
            'split': lambda m: (m[0][3], m[1])
        }

        # Update model meta
        model_meta[se_resnext101] = meta


    def get_model(self):
        """Returns the model to be passed to a Learner.
        """
        return self.model


class EfficientNet(object):
    def __init__(self):
        # Wrap model for PyTorch API
        def efficient_net_b0(pretrained=True):
            model = EfficientNet.from_pretrained('efficientnet-b0')

            return nn.Sequential(model)

        self.model = efficient_net_b0

        # Define splits and layer groups
        meta = {
            'cut': noop,
            'split': lambda m: (list(m[0][0].children())[2][7], m[1])
        }

        # Update model meta
        model_meta[efficient_net_b0] = meta

        # Requires a custom head to adapt the output size to our task.
        self.custom_head = nn.Linear(1000, 196)


    def get_model(self):
        """Returns the model to be passed to a Learner.
        """
        return self.model


    def get_custom_head(self):
        return self.custom_head
