"""Class that ensembles multiple models together to make a prediction.
"""
import numpy as np
from fastai.basic_data import DatasetType

from src.configs.constants import SAVED_DIR


class Ensemble(object):
    def __init__(self, learners):
        """
        Args:
            learners (list[Learner]): List of loaded Learners.
                All Learners must have the same loaded test dataset.
        """
        self.learners = learners


    def predict(self, tta):
        """Get predictions from the test dataset of the Learners,
        then perform soft voting.

        Args:
            tta (boolean): Whether to perform test-time augmentation.

        Returns:
            np.array: of shape (len(data), n_classes)
        """
        # Get size of dataset
        n_x = len(self.learners[0].data.test_ds)
        # Get number of classes
        n_classes = len(learn.data.classes)

        # Init prediction array
        overall_preds = np.zeros(n_x, n_classes)

        for learn in self.learners:
            # Get predictions
            if tta:
                preds, _ = learn.TTA(ds_type=DatasetType.Test)
            else:
                preds, _ = learn.get_preds(ds_type=DatasetType.Test)
            # Add to overall predictions
            overall_preds += preds

        # Average probability scores
        return overall_preds / len(self.learners)
