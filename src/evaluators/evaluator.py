"""Object that is responsible for model evaluation (and inference)
on a test dataset.

Takes in a saved Learner file.
Loads test data:
    - Images from 'test/' folder
    - Labels from 'test_labels.csv'
"""
import os
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from fastai.basic_data import DatasetType
from fastai.basic_train import load_learner
from fastai.vision import ImageList

from src.configs.constants import (
    SAVED_DIR, DATA_DIR, TEST_FOLDER, TEST_DF_NAME, IMG_COL, CLASS_COL,
    TEST_LOG_PATH
)


class Evaluator(object):
    def __init__(self, learn_name, tta, exp_name):
        """Logs test info to csv file after initialization.

        Args:
            learn_name (str): Name of the saved Learner file,
                loads from f'saved/{learn_name}.pkl'
            tta (boolean): Whether to perform test time augmentation.
            exp_name (str): Experiment name for logging.
        """
        self.exp_name = exp_name

        # Initialize test ImageList
        test_imgs = ImageList.from_csv(
            path=DATA_DIR,
            folder=TEST_FOLDER,
            csv_name=TEST_DF_NAME,
            cols=IMG_COL
        )

        # Initialize Learner from test data
        self.learn = load_learner(
            path=SAVED_DIR,
            file=f'{learn_name}.pkl',
            test=test_imgs,
        )

        # Get classes list
        self.classes = self.learn.data.classes

        # Initialize ground truth labels
        self._init_labels()

        # Get probability scores from model
        if tta:
            self.y_prob, _ = self.learn.get_preds(ds_type=DatasetType.Test)
        else:
            self.y_prob, _ = self.learn.TTA(ds_type=DatasetType.Test)

        # Extract predicted labels from probability scores
        self.y_pred = np.argmax(self.y_prob, axis=1)

        # Compute metrics
        self._init_metrics()

        # Log test info
        self._log_info()


    def get_classes(self):
        """Retrieve list of classes.
        The integer label of the class can be retrieved by
        self.get_classes.index(class)

        Returns:
            list: of length 196.
        """
        return self.classes


    def get_prob_scores(self):
        """Retrieve probability scores.

        Returns:
            np.array of shape (n_images, n_classes)
                         i.e. (n_images, 196)
        """
        return self.y_prob


    def get_preds(self):
        """Retrieve predicted labels.

        Returns:
            np.array: of integer labels.
        """
        return self.y_pred


    def get_metrics(self):
        """
        Returns:
            (accuracy, precision, recall, fscore)
        """
        return self.accuracy, self.precision, self.recall, self.fscore


    def _init_metrics(self):
        """Evaluates the model on the data by computing metrics
        like accuracy, precision, recall, fscore, etc.
        """
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision, self.recall, self.fscore, _ = \
            precision_recall_fscore_support(self.y_true, self.y_pred, average='micro')


    def _init_labels(self):
        """Retrieve ground truth labels from the test dataframe.
        """
        test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_DF_NAME))
        y_true = test_df[CLASS_COL].values
        self.y_true = [self.classes.index(y) for y in y_true]


    def _log_info(self):
        fieldnames = [
            'exp_name',
            'accuracy',
            'precision',
            'recall',
            'fscore',
            'remarks',
        ]

        # Fill up information
        row = {}
        row['exp_name'] = self.exp_name
        row['accuracy'] = self.accuracy
        row['precision'] = self.precision
        row['recall'] = self.recall
        row['fscore'] = self.fscore
        row['remarks'] = ''

        with open(TEST_LOG_PATH, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(row)
