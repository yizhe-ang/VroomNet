import os

DATA_DIR = 'data'
DATA_PATH = os.path.join(DATA_DIR, 'data.csv')
LABELS_PATH = os.path.join(DATA_DIR, 'label_map.json')

# Column names of data features
IMG_COL = 'img_path'
CLASS_COL = 'class'

N_CLASSES = 196

MODEL_DIR = 'models'
MODEL_CSV_DIR = os.path.join(MODEL_DIR, 'model_csv')
MODEL_LOGS_DIR = os.path.join(MODEL_DIR, 'model_logs')
MODEL_WEIGHTS_DIR = os.path.join(MODEL_DIR, 'model_weights')
EXPS_PATH = os.path.join(MODEL_DIR, 'exps_info.csv')
