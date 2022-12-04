import os, sys, glob
import shelve
import json
import cv2
import numpy as np
import imageio
from tqdm import tqdm
import copy
import open3d as o3d
import yaml
import time

treg = o3d.t.pipelines.registration 
import open3d as o3d
import numpy as np 
from scipy.spatial import cKDTree
from tqdm import tqdm
import shutil


ood_root = os.environ['OOD_ROOT']
dataset_root = os.path.join(ood_root , 'ours/data3/data3_syn_raw/')
new_dataset_root = os.path.join(ood_root, 'ours/data3/data3_syn_source/')


def i2s(num):
    return "{0:06d}".format(num)


scene_ids_per_gripper = {}

scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
for scene_id in tqdm(scene_ids):

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    robot_info_path = os.path.join(scene_folder_path, "robot_info.json")
    with open(robot_info_path, 'r') as j_file:
        robot_info = json.load(j_file)
    if robot_info['gripper_type'] not in scene_ids_per_gripper:
        scene_ids_per_gripper[robot_info['gripper_type']] = [scene_id]
    else:
        scene_ids_per_gripper[robot_info['gripper_type']].append(scene_id)
new_scene_id = 1

for gripper in scene_ids_per_gripper.keys():
    n_scenes = 0
    for scene_id in tqdm(scene_ids_per_gripper[gripper]):
        n_rgb_imgs = len(glob.glob(os.path.join(dataset_root, i2s(scene_id), "rgb", "*.png")))
        if n_rgb_imgs < 150:
            print("Skipping scene {} because it has only {} rgb images".format(scene_id, n_rgb_imgs))
            continue
        n_scenes += 1
        if n_scenes > 23:
            break
        scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
        new_scene_folder_path = os.path.join(new_dataset_root, i2s(new_scene_id))
        # rsync 
        os.system("rsync -a {} {}".format(scene_folder_path, new_dataset_root))
        shutil.move(os.path.join(new_dataset_root, i2s(scene_id)), os.path.join(new_dataset_root, i2s(new_scene_id)))
        scene_gt_info_path = os.path.join(scene_folder_path, "scene_gt_{0:06d}.json".format(scene_id))
        new_scene_gt_info_path = os.path.join(new_scene_folder_path, "scene_gt_{0:06d}.json".format(new_scene_id))
        shutil.copy(scene_gt_info_path, new_scene_gt_info_path)
        new_scene_id += 1

