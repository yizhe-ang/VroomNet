"""
Convert the given "cars_annos.mat" data file to csv format.
"""
import csv
import scipy.io
import numpy as np
import json


MAT_PATH = 'data/cars_annos.mat'
LABEL_PATH = 'data/label_map.json'
CSV_PATH = 'data/data.csv'

# Load label map
with open(LABEL_PATH) as json_file:
    label_map = json.load(json_file)

with open(CSV_PATH, 'w') as csvfile:
    mat = scipy.io.loadmat(MAT_PATH)

    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['img_path','class','bbox_x1','bbox_y1','bbox_x2','bbox_y2','test'])

    for annotation in mat['annotations'][0]:
        test = np.squeeze(annotation['test'])
        im_path = str(np.squeeze(annotation['relative_im_path']))

        cls = np.squeeze(annotation['class'])
        cls = label_map[str(cls)]

        x1 = np.squeeze(annotation['bbox_x1'])
        y1 = np.squeeze(annotation['bbox_y1'])
        x2 = np.squeeze(annotation['bbox_x2'])
        y2 = np.squeeze(annotation['bbox_y2'])

        csvwriter.writerow([im_path, cls, x1, y1, x2, y2, test])
