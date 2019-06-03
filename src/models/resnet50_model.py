import os
from keras.applications.resnet50 import ResNet50
from keras.layers import (
    Dense, Activation, Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from src.base.base_model import BaseModel
from src.configs.constants import MODEL_WEIGHTS_DIR, N_CLASSES


class ResNet50Model(BaseModel):
    def __init__(self, config):
        """config.model should specify:

        ln_rate:
        n_freeze:
        weights: either 'imagenet' or name of saved weights.
        """
        super(ResNet50Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        IMG_SIZE = self.config.data_loader.img_size
        LN_RATE = self.config.model.ln_rate
        FREEZE = self.config.model.freeze
        WEIGHTS = self.config.model.weights

        # Import pretrained model
        base_model = ResNet50(weights=WEIGHTS if WEIGHTS=='imagenet' else None,
                              include_top=False,
                              input_shape=(IMG_SIZE, IMG_SIZE, 3),
                              pooling='avg')

        # FREEZE LAYERS HERE ---------------------------------------------------
        if FREEZE:
            for layer in base_model.layers:
                layer.trainable = False

        # Add classifier head
        x = base_model.output
        x = Flatten()(x)

        x = BatchNormalization()(x)
        x = Dropout(rate=0.25)(x)
        x = Dense(512, activation='relu')(x)

        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        predictions = Dense(N_CLASSES, activation='softmax')(x)

        # Instantiate model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # LOAD WEIGHTS HERE ----------------------------------------------------
        if WEIGHTS != 'imagenet':
            self.load(os.path.join(MODEL_WEIGHTS_DIR, f'{WEIGHTS}.h5'))

        # Compile model
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr=LN_RATE),
            # what about precision and recall?
            metrics=[
                'acc',
                # Precision(),
                # Recall()
            ]
        )
