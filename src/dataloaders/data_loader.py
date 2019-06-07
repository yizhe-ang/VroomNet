"""Object that is responsible for the loading of data:
    - Images from 'train/' folder
    - Labels from 'train_labels.csv'
"""
import os
import pandas as pd

from fastai.vision import (
    ImageList,
    get_transforms, imagenet_stats,
)

from src.dataloaders.preprocess import get_indices_split
from src.configs.constants import (
    IMG_COL, CLASS_COL,
    DATA_DIR, TRAIN_DF_NAME, TRAIN_FOLDER
)


class DataLoader(object):
    def __init__(self):
        # Read in the training DataFrame
        df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_DF_NAME))
        # Get stratified split indices
        train_idx, val_idx = get_indices_split(df, CLASS_COL, 0.2)

        # Initialize the augmentation/transformation function.
        self._init_tfms()

        # Initialize the ImageList
        # (source image data and labels before any transformations)
        self.src = (ImageList
            .from_csv(path=DATA_DIR,
                      csv_name=TRAIN_DF_NAME,
                      folder=TRAIN_FOLDER,
                      cols=IMG_COL)
            # Stratified split
            .split_by_idxs(train_idx, val_idx)
            # Get labels
            .label_from_df(CLASS_COL))


    def get_data_bunch(self, img_size=224, batch_size=32):
        """Initializes the DataBunch to be fed into a Learner for training.
        Defines any preprocessing and augmentations for the image data.

        Args:
            img_size (int): Resizes the image to (img_size, img_size).
                Defaults to 224.
            batch_size (int): Batch size. Defaults to 32.

        Returns:
            DataBunch:
        """
        data = (self.src
            .transform(self.tfms, size=img_size)
            .databunch(bs=batch_size)
            # Normalize as per imagenet stats for transfer learning
            .normalize(imagenet_stats))

        return data


    def _init_tfms(self):
        """Initialize the augmentation/transformation function.
        """
        tfms = get_transforms()

        self.tfms = tfms
