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
dataset_root = os.path.join(ood_root, 'ours/data2/data2_real_source/all')


with open(os.path.join(ood_root, "ours/data1/models_notaligned_to_models_aligned.json"), "r") as f:
    tranform_to_aligned = json.load(f)

scene_ids = list(range(1, 1051))


for scene_id in tqdm(scene_ids): 
    scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_{0:06d}.json".format(scene_id))
    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)
    new_scene_gt = {}
    for im_id in scene_gt.keys():
        obj_gts = scene_gt[im_id]
        new_obj_gts = []
        for obj_gt in obj_gts:
            obj_id = obj_gt['obj_id']
            cam_H_m2c = np.eye(4)
            cam_H_m2c[:3, :3] = np.array(obj_gt["cam_R_m2c"]).copy().reshape(3, 3)
            cam_H_m2c[:3, 3] = np.array(obj_gt["cam_t_m2c"]).copy()

            obj_id = obj_gt["obj_id"]
            H_m2m_aligned = np.array(tranform_to_aligned[str(obj_id)]).copy().reshape(4, 4)
            H_m_aligned2m = np.linalg.inv(H_m2m_aligned)
            cam_H_m_aligned2c = np.matmul(cam_H_m2c.copy(), H_m_aligned2m.copy(),)
            print(obj_id, np.sum(np.abs(cam_H_m2c - cam_H_m_aligned2c)))

            cam_R_m_aligned2c = cam_H_m_aligned2c[:3, :3]
            cam_t_m_aligned2c = cam_H_m_aligned2c[:3, 3].reshape(3, 1)
            new_obj_gt = {
                "cam_R_m2c": cam_R_m_aligned2c.reshape(-1).tolist(),
                "cam_t_m2c": cam_t_m_aligned2c.reshape(-1).tolist(),
                "obj_id": obj_id,
                "inst_id": obj_gt["inst_id"],
            }
            new_obj_gts.append(new_obj_gt)
        new_scene_gt[im_id] = new_obj_gts
    print(sorted(list(new_scene_gt.keys()), key=lambda x: int(x)))
    new_scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_{0:06d}.json".format(scene_id))
    with open(new_scene_gt_path, "w") as f:
        json.dump(new_scene_gt, f)