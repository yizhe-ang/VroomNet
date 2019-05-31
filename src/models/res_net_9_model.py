from src.base.base_model import BaseModel
from src.configs.constants import MODEL_WEIGHTS_DIR

from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Layer, BatchNormalization, Activation, Add,
    MaxPool2D, GlobalAvgPool2D, Flatten, Dense, Input
)
from tensorflow.keras.optimizers import Adam
import os


def bottleneck_residual_block(X, kernel_size, filters, reduce=False, s=2):
     # unpack the tuple to retrieve Filters of each CONV layer
     F1, F2 = filters
    
     # Save the input value to use it later to add back to the main path.
     X_shortcut = X
    
     # if condition if reduce is True
     if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides
        X_shortcut = Conv2D(filters = F2, kernel_size = (1, 1), strides = (s,s),
                            padding='same')(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
        
 		# if reduce, we will need to set the strides of the first conv to be similar to the shortcut strides
        X = Conv2D(filters = F1, kernel_size = (3, 3), strides = (s,s), padding = 'same')(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        
     else:
         # First component of main path
         X = Conv2D(filters = F1, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
         X = BatchNormalization(axis = 3)(X)
         X = Activation('relu')(X)
    
     # Second component of main path
     X = Conv2D(filters = F2, kernel_size = kernel_size, strides = (1,1), padding = 'same')(X)
     X = BatchNormalization(axis = 3)(X)
     X = Activation('relu')(X)
  
     # Final step: Add shortcut value to main path, and pass it through a RELU activation
     X = Add()([X, X_shortcut])
     X = Activation('relu')(X)
  
     return X


class ResNet9Model(BaseModel):
    def __init__(self, config):
        """Will load weights if specified.
        """
        super(ResNet9Model, self).__init__(config)
        self.build_model()


    def build_model(self):
        img_dim = self.config.data_loader.img_dim

        X_input = Input((img_dim, img_dim, 3))
  
	    # Stage 1
        X = Conv2D(32, (7, 7), strides=(2, 2), name='conv1', padding='same')(X_input)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)
	  
	    # Stage 2
        X = bottleneck_residual_block(X, 3, [32, 32])
	  
	    # Stage 3
        X = bottleneck_residual_block(X, 3, [64, 64], reduce=True, s=2)
	  
	    # Stage 4
        X = bottleneck_residual_block(X, 3, [128, 128], reduce=True, s=2)
	  
	    # Stage 5
        X = bottleneck_residual_block(X, 3, [256, 256], reduce=True, s=2)

        X = GlobalAvgPool2D()(X)
	  
	    # output layer
        X = Flatten()(X)
        X = Dense(5, activation='softmax', name='fc')(X)
    
	    # Create the model
        self.model = Model(inputs = X_input, outputs = X, name='ResNet9')
	  
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr=self.config.model.learning_rate),
            metrics=['acc'],
        )

        # Load weights if specified
        weights = self.config.model.weights
        if weights != 'None':
            self.load(os.path.join(MODEL_WEIGHTS_DIR, f'{weights}.h5'))
