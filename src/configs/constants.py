import os

DATA_DIR = 'data'
# DATA_DIR = '/floyd/input/data'
TRAIN_DF_NAME = 'train_labels.csv'
TEST_DF_NAME = 'test_labels.csv'

TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'

SAVED_DIR = 'saved'
WEIGHTS_FOLDER = 'model_weights'
CSV_FOLDER = 'model_csv'

TRAIN_LOG_PATH = os.path.join(SAVED_DIR, 'train_info.csv')
TEST_LOG_PATH = os.path.join(SAVED_DIR, 'test_info.csv')

# Column names of data features
IMG_COL = 'img_name'
CLASS_COL = 'class'
