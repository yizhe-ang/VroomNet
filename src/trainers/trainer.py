"""Object that is responsible for model training.

Takes in a model and a DataBunch, and initializes a Learner.
"""
import os

from fastai.metrics import accuracy
from fastai.callbacks.csv_logger import CSVLogger
from fastai.callbacks.tracker import SaveModelCallback
from fastai.vision import cnn_learner

from src.configs.constants import (
    SAVED_DIR, WEIGHTS_FOLDER, CSV_FOLDER
)


class Trainer(object):
    def __init__(self, model, data_bunch, exp_name):
        """
        Args:
            model: Model that makes up the backbone of our
                classification model.
            data_bunch (DataBunch): Data to be trained on.
            exp_name (str): Name of this training experiment.
        """
        self.exp_name = exp_name
        # Initalize metrics
        self._init_metrics()

        # Initialize Learner
        self.learn = cnn_learner(
            data=data_bunch,
            base_arch=model,
            metrics=self.metrics,
            path=SAVED_DIR,
            model_dir=WEIGHTS_FOLDER
        )

        # Initialize callbacks
        self._init_callbacks()


    def lr_find(self, freeze):
        """Plots the learning rate finder.

        Args:
            freeze (boolean): Specify whether the model layers are to be
                freezed or unfreezed during training, i.e.
                True: To train classifier head.
                False: To finetune entire model.
        """
        if freeze:
            self.learn.freeze()
        else:
            self.learn.unfreeze()

        self.learn.lr_find()
        self.learn.recorder.plot(suggestion=True)


    def train(self, n_epochs, max_lr, freeze):
        """Fits the data using the 1cycle policy.
        Will load best weights (that yields lowest validation loss) after training.

        Args:
            n_epochs (int): Number of epochs to train.

            max_lr (float/slice): Learning rate/s.
                float: Same learning rate for all layers.
                slice: Discriminative layer training.

            freeze (boolean): Specify whether the model layers are to be
                freezed or unfreezed during training, i.e.
                True: To train classifier head.
                False: To finetune entire model.
        """
        if freeze:
            self.learn.freeze()
        else:
            self.learn.unfreeze()

        self.learn.fit_one_cycle(
            cyc_len=n_epochs,
            max_lr=max_lr,
            callbacks=self.callbacks,
        )


    def save_weights(self, suffix):
        """Save the current model weights from the Learner.

        Args:
            suffix (str): Saved path will be
                f'saved/model_weights/{exp_name}_{suffix}.pth'
        """
        self.learn.save(f'{self.exp_name}_{suffix}')


    def load_weights(self, suffix):
        """Load model weights into the Learner.

        Args:
            suffix (str): Loaded path will be
                f'saved/model_weights/{exp_name}_{suffix}.pth'
        """
        self.learn.load(f'{self.exp_name}_{suffix}')


    def plot_losses(self):
        """Plot train and valid losses.

        How does it record the losses?
        (i.e. across different training stages)
        """
        self.learn.plot_losses()


    def set_data_bunch(self, data_bunch):
        """Re-sets the DataBunch for the Learner,
        for e.g. when you want to increase the image size.
        """
        self.learn.data = data_bunch


    def export(self):
        """Exports the Learner, to then be used for inference/evaluation.
        """
        self.learn.export(f'{EXP_NAME}.pkl')


    def log_exp_info(self):
        """Logs the information for the experiment.
        Log for each training stage?
        """
        pass


    def _init_metrics(self):
        """Initalize metrics to be calculated on validation data
        during training.
        """
        self.metrics = [accuracy]


    def _init_callbacks(self):
        """Initialize callbacks.

        Elaborate on callbacks.
        """
        self.callbacks = []

        # Should we save a different csv file for each training stage?
        self.callbacks.append(CSVLogger(
            learn=self.learn,
            append=True,
            filename=os.path.join(CSV_FOLDER, self.exp_name)
        ))

        self.callbacks.append(SaveModelCallback(
            learn=self.learn,
            # Also loads best model at the end of training
            every='improvement',
            name=self.exp_name,
        ))
