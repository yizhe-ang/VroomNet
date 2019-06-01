import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (
    Dense, Activation, Flatten, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from src.base.base_model import BaseModel
from src.configs.constants import MODEL_WEIGHTS_DIR, N_CLASSES


class ResNet50Model(BaseModel):
    def __init__(self, config):
        super(ResNet50Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        IMG_SIZE = self.config.data_loader.img_size
        LN_RATE = self.config.model.ln_rate

        # Import pretrained model
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(IMG_SIZE, IMG_SIZE, 3),
                              pooling='avg')

        # Freeze all pretrained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add classifier layer
        x = base_model.output
        x = Flatten()(x)
        predictions = Dense(N_CLASSES, activation='softmax')(x)

        # Instantiate model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr=LN_RATE),
            # what about precision and recall?
            metrics=[
                'acc',
                # Precision(), Recall()
            ]
        )

        # Load weights if specified
        # weights = self.config.model.weights
        # if weights != 'None':
        #     self.load(os.path.join(MODEL_WEIGHTS_DIR, f'{weights}.h5'))
