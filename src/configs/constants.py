import os

DATA_DIR = 'data'
# DATA_DIR = '/floyd/input/data'
DATA_PATH = os.path.join(DATA_DIR, 'data.csv')
LABELS_PATH = os.path.join(DATA_DIR, 'label_map.json')

# Column names of data features
IMG_COL = 'img_path'
CLASS_COL = 'class'

N_CLASSES = 196

MODEL_DIR = 'saved'
# MODEL_CSV_DIR = os.path.join(MODEL_DIR, 'model_csv')
MODEL_CSV_DIR = MODEL_DIR
# MODEL_LOGS_DIR = os.path.join(MODEL_DIR, 'model_logs')
MODEL_LOGS_DIR = MODEL_DIR
# MODEL_WEIGHTS_DIR = os.path.join(MODEL_DIR, 'model_weights')
MODEL_WEIGHTS_DIR = MODEL_DIR
EXPS_PATH = os.path.join(MODEL_DIR, 'exps_info.csv')
