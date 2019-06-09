import pretrainedmodels

# Change to return a Dictionary!!!!!
class SEResNeXt101(object):
    def get_model(self):
        def se_resnext101(pretrained=False):
            pretrained = 'imagenet' if pretrained else None
            model = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)

            return model

        return se_resnext101


    def get_meta(self):
        return { 'cut': -2,
                 'split': lambda m: (m[0][3], m[1]) }
