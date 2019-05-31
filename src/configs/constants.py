import os

DATA_DIR = 'data'
IMG_DIR = os.path.join(DATA_DIR, 'images')
DATA_PATH = os.path.join(DATA_DIR, 'bounding_box_labels.csv')

MODEL_DIR = 'models'
MODEL_CSV_DIR = os.path.join(MODEL_DIR, 'model_csv')
MODEL_LOGS_DIR = os.path.join(MODEL_DIR, 'model_logs')
MODEL_WEIGHTS_DIR = os.path.join(MODEL_DIR, 'model_weights')
EXPS_PATH = os.path.join(MODEL_DIR, 'exps_info.csv')
