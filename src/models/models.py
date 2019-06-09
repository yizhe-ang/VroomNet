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


def se_resnet50(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnet(pretrained=pretrained)

    return model

def senet_split(m):
    return (m[0][3], m[1])

learn = cnn_learner(data_bunch, se_resnet50, pretrained=True,
                    cut=-2, senet_split)
