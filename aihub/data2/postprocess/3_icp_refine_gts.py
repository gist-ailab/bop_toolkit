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
    scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_c1p1_{0:06d}.json".format(scene_id))
    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)
    scene_camera_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_camera.json")
    with open(scene_camera_path, "r") as f:
        scene_camera = json.load(f)
    
    new_scene_gt = {}
    for im_id in im_ids:
        im_gts = scene_gt[str(im_id)]
        new_im_gts = []
        for obj_gt in im_gts:
            obj_id = obj_gt['obj_id']
            inst_id = obj_gt['inst_id']
            
            H_c2m = np.eye(4)
            H_c2m[:3, :3] = np.array(obj_gt['cam_R_m2c']).copy().reshape(3, 3)
            H_c2m[:3, 3] = np.array(obj_gt['cam_t_m2c']).copy()

            scene_pcd = o3d.io.read_point_cloud(os.path.join(dataset_root, f"{scene_id:06d}", "pcd", f"{im_id:06d}.pcd"))
            object_model_path = os.path.join(ood_root, f"ours/data1/models/obj_{obj_id:06d}.ply")
            object_pcd = o3d.io.read_point_cloud(object_model_path)
            object_pcd.transform(H_c2m)
            object_pcd.scale(0.001, [0, 0, 0])
            cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            object_pcd = object_pcd.select_by_index(ind)
            # o3d.visualization.draw_geometries([scene_pcd, object_pcd])

            trans_init = np.identity(4)
            threshold = 0.004
            reg = o3d.pipelines.registration.registration_icp(object_pcd, scene_pcd, threshold, trans_init,
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                max_iteration=50))
            delta_H = reg.transformation.copy()              
            new_obj_gt = copy.deepcopy(obj_gt)   
            print(np.sum(np.abs(reg.transformation[:3, 3])))                              
            if np.sum(np.abs(reg.transformation[:3, 3])) < 0.25:
                delta_H[:3, 3] = delta_H[:3, 3] * 1000
                H_c2m = np.matmul(delta_H, H_c2m)
                new_obj_gt['cam_R_m2c'] = H_c2m[:3, :3].reshape(-1).tolist()
                new_obj_gt['cam_t_m2c'] = H_c2m[:3, 3].tolist()
                print("refine scene_id: {}, im_id: {}, obj_id: {}, inst_id: {}".format(scene_id, im_id, obj_id, inst_id))
            new_im_gts.append(new_obj_gt)
        new_scene_gt[str(im_id)] = new_im_gts
    new_scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_icp_{0:06d}.json".format(scene_id))
    with open(new_scene_gt_path, "w") as f:
        json.dump(new_scene_gt, f)



