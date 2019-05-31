from src.base.base_model import BaseModel
from src.configs.constants import MODEL_WEIGHTS_DIR

from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Layer, BatchNormalization, Activation,
    MaxPool2D, GlobalAvgPool2D, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
import os

# ------------------------------------------------------------------------------
# Create the Residual Unit
# ------------------------------------------------------------------------------
DefaultConv2D = partial(Conv2D, kernel_size=3, strides=1,
                        padding='SAME', use_bias=False)

class ResidualUnit(Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)

        self.activation = tf.keras.activations.get(activation)

        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)

        skip_Z = inputs

        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z)


# ------------------------------------------------------------------------------
# Full ResNet Here
# ------------------------------------------------------------------------------
class ResNetModel(BaseModel):
    def __init__(self, config):
        """Will load weights if specified.

        config.model will contain:

        planes: Number of Residual Units for each filter size.
            e.g. [1, 1, 1, 1] corresponds to ResNet-9,
                 [3, 4, 6, 3] corresponds to ResNet-34.

        filter_sizes: Specify the four different sizes of filters.
            e.g. [64, 128, 256, 512]
        """
        super(ResNetModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        n_1, n_2, n_3, n_4 = self.config.model.planes
        f_1, f_2, f_3, f_4 = self.config.model.filter_sizes
        img_dim = self.config.data_loader.img_dim

        self.model = Sequential()
        self.model.add(DefaultConv2D(f_1, kernel_size=7, strides=2,
                                input_shape=(img_dim, img_dim, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=3, strides=2, padding='SAME'))

        prev_filters = f_1

        for filters in [f_1]*n_1 + [f_2]*n_2 + [f_3]*n_3 + [f_4]*n_4:
            strides = 1 if filters == prev_filters else 2
            self.model.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters

        self.model.add(GlobalAvgPool2D())
        self.model.add(Flatten())
        self.model.add(Dense(5, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr=self.config.model.learning_rate),
            metrics=['acc'],
        )

        # Load weights if specified
        weights = self.config.model.weights
        if weights != 'None':
            self.load(os.path.join(MODEL_WEIGHTS_DIR, f'{weights}.h5'))
