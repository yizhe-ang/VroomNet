"""Object that is responsible for model training.

Takes in a model and a DataBunch, and initializes a Learner.
"""
import os
import csv
import pandas as pd

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


    def train(self, n_epochs, max_lr, freeze, name):
        """Fits the data using the 1cycle policy.

        Logs per-epoch metrics to csv file,
        and saves the best weight (that yields lowest validation loss).
        Will load best weights at the end of training.
        Also logs training stage information at the end of training.

        Args:
            n_epochs (int): Number of epochs to train.

            max_lr (float/slice): Learning rate/s.
                float: Same learning rate for all layers.
                slice: Discriminative layer training.

            freeze (boolean): Specify whether the model layers are to be
                freezed or unfreezed during training, i.e.
                True: To train classifier head.
                False: To finetune entire model.

            name (str): Name of training stage (used to save csv and weights).
                csv saved in: f'saved/model_csv/{exp_name}_{name}.csv'
                model weights saved in: f'saved/model_weights/{exp_name}_{name}.pth'
        """
        if freeze:
            self.learn.freeze()
        else:
            self.learn.unfreeze()

        # Get callbacks
        callbacks = self._get_callbacks(name)

        self.learn.fit_one_cycle(
            cyc_len=n_epochs,
            max_lr=max_lr,
            callbacks=callbacks,
        )

        # Log training stage info
        self._log_info(n_epochs, max_lr, freeze, name)


    def save_weights(self, name):
        """Save the current model weights from the Learner.

        Args:
            name (str): Saved path will be
                f'saved/model_weights/{exp_name}_{name}.pth'
        """
        self.learn.save(f'{self.exp_name}_{name}')


    def load_weights(self, name):
        """Load model weights into the Learner.

        Args:
            name (str): Loaded path will be
                f'saved/model_weights/{exp_name}_{name}.pth'
        """
        self.learn.load(f'{self.exp_name}_{name}')


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
        self.learn.export(f'{self.exp_name}.pkl')


    def _log_info(self, n_epochs, max_lr, freeze, name):
        """Logs the information after each training stage.
        Extract from CSVLogger.
        """
        fieldnames = [
            'name',
            'epoch',
            'train_loss',
            'valid_loss',
            'accuracy',
            'max_lr',
            'freeze',
            'img_size',
            'remarks',
        ]

        # Open training logged csv file
        csv_path = os.path.join(SAVED_DIR, CSV_FOLDER, f'{self.exp_name}_{name}.csv')
        df = pd.read_csv(csv_path)

        # Grab the row with the lowest validation loss
        idx = df['valid_loss'].idxmin()
        row = df.loc[idx].to_dict()

        # Get training image size
        img_size = self.learn.data.train_ds[0][0].shape[1]

        # Fill up information
        del row['time']
        row['epoch'] += 1
        row['name'] = f'{self.exp_name}_{name}'
        row['max_lr'] = str(max_lr)
        row['freeze'] = freeze
        row['img_size'] = img_size
        row['remarks'] = ''

        with open(EXPS_PATH, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(row)


    def _init_metrics(self):
        """Initalize metrics to be calculated on validation data
        during training.
        """
        self.metrics = [accuracy]


    def _get_callbacks(self, name):
        """Retrieve callbacks to be used for training.

        Args:
            name (str): Name of training stage (used to save files).
                csv saved in: f'saved/model_csv/{exp_name}_{name}.csv'
                model weights saved in: f'saved/model_weights/{exp_name}_{name}.pth'

        Returns:
            List of Callbacks.
        """
        callbacks = []

        # Logs metrics for each training stage
        callbacks.append(CSVLogger(
            learn=self.learn,
            append=False,
            filename=os.path.join(CSV_FOLDER, f'{self.exp_name}_{name}')
        ))

        # Saves the best model weights
        callbacks.append(SaveModelCallback(
            learn=self.learn,
            # Also loads best model weights at the end of training
            every='improvement',
            name=f'{self.exp_name}_{name}',
        ))

        return callbacks
