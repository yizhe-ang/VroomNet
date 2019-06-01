"""
Extracts the label map from the given "cars_annos.mat" data file into a JSON file.
"""
import scipy.io as sio
import json


MAT_PATH = 'data/cars_annos.mat'
LABEL_PATH = 'data/label_map.json'

mat = sio.loadmat(MAT_PATH)

label_map = {}

for i, vehicle_class in enumerate(mat['class_names'][0]):
    print(i+1, str(vehicle_class[0]))
    label_map[i+1] = str(vehicle_class[0])

with open(LABEL_PATH, 'w') as json_file:
    json.dump(label_map, json_file)
