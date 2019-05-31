from src.base.base_model import BaseModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Activation
)

DefaultConv2D = partial(Conv2D, kernel_size=3, padding='same')

class VGG(BaseModel):
    def __init__(self, config):
        """config.model will contain:

        batch_norm (boolean): Whether to include batch norm.

        dropout_rate (float): Dropout rate.

        conv_cfg: Configuration for Conv Layers.
            e.g. [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            corresponds to VGG-11.

        fc_cfg: Configuration for FC Layers.
            e.g. [4096, 4096, 4096]
        """
        super(VGG, self).__init__(config)
        self.build_model()

    def build_model(self):
        img_dim = self.config.data_loader.img_dim
        conv_cfg = self.config.model.conv_cfg
        fc_cfg = self.config.model.fc_cfg

        self.model = Sequential()

        # Set up conv layers
        for i, l in enumerate(conv_cfg):
            if l == 'M':
                self.model.add(MaxPool2D(pool_size=2, strides=2))
            else:
                # If first layer
                if i == 1:
                    self.model.add(DefaultConv2D(
                        filters=l, input_shape=(img_dim, img_dim, 3)
                    ))
                else:
                    self.model.add(DefaultConv2D(filters=l))

                # If batch norm
                if batch_norm:
                    self.model.add(BatchNormalization())

                self.model.add(Activation('relu'))

        self.model.add(Flatten())

        # Set up fc layers
        for l in fc_cfg:
            self.model.add(Dense(l, activation='relu'))
            # Dropout?
            # self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(5, activation='softmax'))
