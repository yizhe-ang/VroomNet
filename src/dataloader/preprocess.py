"""
Module that contains any functions relating to preprocessing, data loading,
and the loading and manipulation of images.
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import (
    img_to_array, load_img
)


# Creates the indices for Train/Val split on given dataset.
def get_indices_split(data, col, val_prop=0.3):
    """Performs Stratified Sampling based on class proportions.
    (with random seed set).

    Args:
        data (pd.DataFrame): 
        col (str): The feature/label to stratify on.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_prop, random_state=42)

    train_indices, val_indices = next(split.split(data, data[col]))

    return train_indices, val_indices


# Loads and crops image from raw data information.
def preprocess_image(row, img_dir):
    """
    Args:
        row (pd.Series): Raw row data information of that image.
        img_dir (str): Directory of image.

    Returns: Tensor of the image.
    """
    img = load_img(os.path.join(img_dir, row['filename']))
    img = img_to_array(img, dtype=np.uint8)

    img = crop_to_bounding_box(
        img,
        row['y'],
        row['x'],
        row['height'],
        row['width']
    )

    return img
