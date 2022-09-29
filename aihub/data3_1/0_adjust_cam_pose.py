import os
import glob
import json
import numpy as np
from tqdm import tqdm
import json

dataset_root = "/home/seung/OccludedObjectDataset/ours/data3/data3_1_raw"


def i2s(num):
    return "{0:06d}".format(num)


scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])

for scene_id in tqdm(scene_ids):

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    if not os.path.isdir(scene_folder_path):
        continue
    print("Processing {}".format(scene_folder_path))
    scene_number = os.path.basename(scene_folder_path)
    scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    with open(scene_camera_info_path, 'r') as j_file:
        scene_camera_info = json.load(j_file)

    for image_number in scene_camera_info.keys():
        cam_R_w2c = np.array(scene_camera_info[image_number]["cam_R_w2c"] ).reshape(3,3)
        cam_t_w2c = np.array(scene_camera_info[image_number]["cam_t_w2c"] )
        se3 = np.eye(4)
        se3[:3,:3] = cam_R_w2c
        se3[:3,3] = cam_t_w2c * 0.001

        mat = np.array([[ 1.   ,  0.   ,  0.   , -0.829],
       [ 0.   ,  1.   ,  0.   , -0.1  ],
       [ 0.   ,  0.   ,  1.   ,  0.405],
       [ 0.   ,  0.   ,  0.   ,  1.   ]])

        se3 = np.matmul(mat, se3)
        cam_R_w2c = se3[:3,:3].reshape(9).tolist()
        se3[:3,3] = se3[:3,3] * 1000
        cam_t_w2c = se3[:3,3].tolist()
        scene_camera_info[image_number]["cam_R_w2c"] = cam_R_w2c
        scene_camera_info[image_number]["cam_t_w2c"] = cam_t_w2c
    
    with open(scene_camera_info_path, 'w') as j_file:
        json.dump(scene_camera_info, j_file, indent=2)
