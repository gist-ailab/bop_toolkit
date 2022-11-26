import os, sys, glob
import json
import cv2
import numpy as np
from tqdm import tqdm
import copy
import open3d as o3d
import pandas as pd

treg = o3d.t.pipelines.registration 
import open3d as o3d
import numpy as np 
from tqdm import tqdm
import shutil


ood_root = os.environ['OOD_ROOT']
dataset_root = os.path.join(ood_root, 'ours/data2/data2_real_source/all')

scene_ids = list(range(1, 11))
im_ids = list(range(1, 53))

for scene_id in tqdm(scene_ids):
    scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_{0:06d}.json".format(scene_id))
    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)
    scene_camera_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_camera.json")
    with open(scene_camera_path, "r") as f:
        scene_camera = json.load(f)
    
    # read all im_gts, and if there are ommited gts in im_id 1, then add them to im_gts
    obj_inst_ids = []
    im_gts_cam1_pose1 = scene_gt[str(1)]
    for im_gt in im_gts_cam1_pose1:
        obj_inst_ids.append("{}_{}".format(im_gt['obj_id'], im_gt['inst_id']))

    H_w2c1p1 = np.eye(4)
    H_w2c1p1[:3, :3] = np.array(scene_camera['1']['cam_R_w2c']).reshape(3, 3)
    H_w2c1p1[:3, 3] = np.array(scene_camera['1']['cam_t_w2c']).reshape(3)

    for im_id in im_ids[1:]:
        im_gts = scene_gt[str(im_id)]
        for im_gt in im_gts:
            obj_inst_id = "{}_{}".format(im_gt['obj_id'], im_gt['inst_id'])
            if obj_inst_id not in obj_inst_ids:
                print("add scene_id {} obj {}, inst {} from {} ".format(scene_id, im_gt['obj_id'], im_gt['inst_id'], im_id))
                H_cthis2m = np.eye(4)
                H_cthis2m[:3, :3] = np.array(im_gt['cam_R_m2c']).copy().reshape(3, 3)
                H_cthis2m[:3, 3] = np.array(im_gt['cam_t_m2c']).copy().reshape(3)

                H_w2cthis = np.eye(4)
                H_w2cthis[:3, :3] = np.array(scene_camera[str(im_id)]['cam_R_w2c']).copy().reshape(3, 3)
                H_w2cthis[:3, 3] = np.array(scene_camera[str(im_id)]['cam_t_w2c']).copy().reshape(3)
                H_c1p12cthis = np.matmul(np.linalg.inv(H_w2c1p1.copy()), H_w2cthis)
                H_c1p12m = np.matmul(H_c1p12cthis, H_cthis2m)
                new_im_gt = {
                    'obj_id': im_gt['obj_id'],
                    'inst_id': im_gt['inst_id'],
                    'cam_R_m2c': H_c1p12m[:3, :3].reshape(-1).tolist(),
                    'cam_t_m2c': H_c1p12m[:3, 3].reshape(-1).tolist(),
                }
                im_gts_cam1_pose1.append(im_gt)
                obj_inst_ids.append(obj_inst_id)

    
    # propagate the gts to all im_ids
    new_scene_gt = {}
    for im_id in im_ids:
        new_im_gts = []
        for im_gt_c1p1 in im_gts_cam1_pose1:
            H_c1p12m = np.eye(4)
            H_c1p12m[:3, :3] = np.array(im_gt_c1p1['cam_R_m2c']).copy().reshape(3, 3)
            H_c1p12m[:3, 3] = np.array(im_gt_c1p1['cam_t_m2c']).copy().reshape(3)

            H_w2cthis = np.eye(4)
            H_w2cthis[:3, :3] = np.array(scene_camera[str(im_id)]['cam_R_w2c']).copy().reshape(3, 3)
            H_w2cthis[:3, 3] = np.array(scene_camera[str(im_id)]['cam_t_w2c']).copy().reshape(3)
            H_thisc2c1p1 = np.matmul(np.linalg.inv(H_w2cthis.copy()), H_w2c1p1)
            H_thisc2m = np.matmul(H_thisc2c1p1, H_c1p12m)
            new_im_gt = {
                'obj_id': im_gt_c1p1['obj_id'],
                'inst_id': im_gt_c1p1['inst_id'],
                'cam_R_m2c': H_thisc2m[:3, :3].reshape(-1).tolist(),
                'cam_t_m2c': H_thisc2m[:3, 3].reshape(-1).tolist(),
            }
            new_im_gts.append(new_im_gt)
        new_scene_gt[str(im_id)] = new_im_gts
    new_scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_c1p1_{0:06d}.json".format(scene_id))
    with open(new_scene_gt_path, "w") as f:
        json.dump(new_scene_gt, f)

    
