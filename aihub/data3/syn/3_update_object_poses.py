import os, sys, glob
import json
import cv2
import numpy as np
from tqdm import tqdm
import copy
import open3d as o3d
import pandas as pd

import open3d as o3d
import numpy as np 
from tqdm import tqdm
import shutil


ood_root = os.environ['OOD_ROOT']
dataset_root = os.path.join(ood_root, 'ours/data3/data3_syn_source/')


with open(os.path.join(ood_root, "ours/data1/models_notaligned_to_models_aligned.json"), "r") as f:
    tranform_to_aligned = json.load(f)

scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
for scene_id in tqdm(scene_ids): 


    scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_{0:06d}.json".format(scene_id))
    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)
    new_scene_gt = {}

    scene_camera_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_camera.json")
    with open(scene_camera_path, "r") as f:
        scene_camera = json.load(f)
    for im_id in scene_gt.keys():
        if int(im_id) > 250:
            continue
        obj_gts = scene_gt[im_id]
        new_obj_gts = []
        for obj_gt in obj_gts:
            obj_id = obj_gt['object_id']
            H_w_to_m = np.eye(4)
            H_w_to_m[:3, :3] = np.array(obj_gt["cam_R_m2c"]).copy().reshape(3, 3)
            H_w_to_m[:3, 3] = np.array(obj_gt["cam_t_m2c"]).copy() * 1000

            H_w_to_c = np.eye(4)
            H_w_to_c[:3, :3] = np.array(scene_camera[im_id]["cam_R_w2c"]).copy().reshape(3, 3)
            H_w_to_c[:3, 3] = np.array(scene_camera[im_id]["cam_t_w2c"]).copy() * 1000

            H_c_to_m = np.linalg.inv(H_w_to_c) @ H_w_to_m

            obj_id = obj_gt['object_id']
            H_m2m_aligned = np.array(tranform_to_aligned[str(obj_id)]).copy().reshape(4, 4)
            H_m_aligned2m = np.linalg.inv(H_m2m_aligned)
            cam_H_m_aligned2c = np.matmul(H_c_to_m.copy(), H_m_aligned2m.copy())
            cam_H_m_aligned2c[1, :] = -cam_H_m_aligned2c[1, :]
            cam_H_m_aligned2c[2, :] = -cam_H_m_aligned2c[2, :]

            cam_R_m_aligned2c = cam_H_m_aligned2c[:3, :3]
            cam_t_m_aligned2c = cam_H_m_aligned2c[:3, 3].reshape(3, 1)
            new_obj_gt = {
                "cam_R_m2c": cam_R_m_aligned2c.reshape(-1).tolist(),
                "cam_t_m2c": cam_t_m_aligned2c.reshape(-1).tolist(),
                "obj_id": obj_id,
                "inst_id": obj_gt["instance_id"],
            }
            new_obj_gts.append(new_obj_gt)
        new_scene_gt[im_id] = new_obj_gts



    new_scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_{0:06d}.json".format(scene_id))
    with open(new_scene_gt_path, "w") as f:
        json.dump(new_scene_gt, f)