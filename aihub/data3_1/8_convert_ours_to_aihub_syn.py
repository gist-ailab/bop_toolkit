from re import I
import shutil
import shutil
from tqdm import tqdm
import os, glob
import pandas as pd
import numpy as np
import json
import cv2
from pycocotools import mask as m

def mask2rle(im):
    im = np.array(im, order='F', dtype=bool)
    rle = m.encode(im)
    rle['counts'] = rle['counts'].decode('ascii')
    rle = rle['counts']
    return rle


dataset_root = "/home/seung/OccludedObjectDataset/ours/data3/data3_1_syn_raw"
aihub_path = "/OccludedObjectDataset/aihub"
# depth
# gt
# pcd
# rgb
# mask 000000_amodal / visible

scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])

for scene_id in tqdm(scene_ids):

    scene_path = os.path.join(dataset_root, "{0:06d}".format(scene_id))
    scene_gt_file = os.path.join(dataset_root, "{0:06d}".format(scene_id), "scene_gt_{0:06d}.json".format(scene_id))

    if not os.path.exists(scene_gt_file):
        print("scene_gt_file not exists: ", scene_gt_file)
        continue
    scene_gt = json.load(open(scene_gt_file, 'r'))
    #!TODO: get this from json
    sub_dir_1, sub_dir_2 =  "UR5", "2-finger(robotiq-2f-140)"
    sub_dir_1_path = os.path.join(aihub_path, '원천데이터', '로봇-물체파지', "가상", sub_dir_1)
    sub_dir_2_path = os.path.join(sub_dir_1_path, sub_dir_2)
    os.makedirs(sub_dir_1_path, exist_ok=True)
    os.makedirs(sub_dir_2_path, exist_ok=True)

    new_scene_path = os.path.join(sub_dir_2_path, "{0:06d}".format(scene_id))
    os.makedirs(new_scene_path, exist_ok=True)
    new_rgb_path = os.path.join(new_scene_path, "rgb")
    os.makedirs(new_rgb_path, exist_ok=True)
    new_depth_path = os.path.join(new_scene_path, "depth")
    os.makedirs(new_depth_path, exist_ok=True)
    new_pcd_path = os.path.join(new_scene_path, "pcd")
    os.makedirs(new_pcd_path, exist_ok=True)
    new_gt_path = os.path.join(new_scene_path, "gt")
    os.makedirs(new_gt_path, exist_ok=True)

    im_ids = sorted([int(x.split(".")[0]) for x in os.listdir(os.path.join(scene_path, "rgb"))])
    if (int(scene_id) + 1) % 10 < 8:
        split = "train"
    elif (int(scene_id) + 1) % 10 == 9 :
        split = "val"
    else:
        split = "test"

    for im_id in tqdm(im_ids):
        aihub_gt = {}
        
        # 1. scene_info
        scene_obj_info_path = scene_path + "/scene_obj_info.json"
        with open(scene_obj_info_path, "r") as f:
            scene_obj_info = json.load(f)
        

        aihub_gt["scene_info"] = {
                "object_set": scene_obj_info[str(im_id)]["object_set"],
                "scene_id": int(scene_id), # 1.2
                "image_id": int(im_id), # 1.3
                "environment": scene_obj_info[str(im_id)]["environment"], # 1.4
                "background": scene_obj_info[str(im_id)]["background"], # 1.5
                "split": split
            }

        scene_camera_path = scene_path + "/scene_camera.json"
        with open(scene_camera_path, "r") as f:
            scene_camera = json.load(f)

        # 2. camera_info
        aihub_gt["camera_info"] = { 
            'cam_R_w2c': scene_camera[str(im_id)]["cam_R_w2c"], # 2.1 
            'cam_t_w2c': scene_camera[str(im_id)]["cam_t_w2c"], # 2.2
            "cam_K": scene_camera[str(im_id)]["cam_K"], # 2.3
            'depth_scale': float(scene_camera[str(im_id)]["depth_scale"]), # 2.4
            "resolution": scene_camera[str(im_id)]["resolution"], # 2.5
            "camera_type": scene_camera[str(im_id)]["camera_type"], # 2.6
            "secs": 0, # 2.7
            "nsecs": 0 # 2.8
        }


        # occlusion order
        occlusion_order = []
        # depth order
        depth_order = []
       
        # 3. annotation
        aihub_gt["annotation"] = []
        if str(im_id) in scene_gt:
            for idx, obj_gt in enumerate(scene_gt[str(im_id)]):
                if not os.path.exists(scene_path + "/mask/{:06d}_amodal.png".format(im_id)):
                    continue
                amodal_mask = cv2.imread(scene_path + "/mask/{:06d}_amodal.png".format(im_id))[:, :, 0]
                visible_mask = cv2.imread(scene_path + "/mask/{:06d}_visible.png".format(im_id))[:, :, 0]
                if np.sum(visible_mask) / np.sum(amodal_mask) > 0.95:
                    invisible_mask = np.zeros_like(amodal_mask)
                else:
                    invisible_mask = cv2.bitwise_xor(amodal_mask, visible_mask)
                target_occlusion_order = [x for x in occlusion_order if int(idx) in [int(x) for x in x["order"].split("&")[0].split("<")]]
                target_depth_order = [x for x in depth_order if int(idx) in [int(x) for x in x["order"].split("<")]]
                if "inst_id" not in obj_gt.keys():
                    obj_gt["inst_id"] = 1
                aihub_gt["annotation"].append({
                    "object_id": obj_gt["object_id"], # 3.1
                    "instance_id": obj_gt["instance_id"], # 3.2
                    "cam_R_m2c": obj_gt["cam_R_m2c"], # 3.3
                    "cam_t_m2c": obj_gt["cam_t_m2c"], # 3.4
                    "visible_mask": mask2rle(visible_mask), # 3.5
                    "invisible_mask": mask2rle(invisible_mask), # 3.6
                    "amodal_mask": mask2rle(amodal_mask), # 3.7
                    "occlusion_order": target_occlusion_order,
                    "depth_order": target_depth_order,
                })
        
        # 4. robot_info
        robot_info_path = scene_path + "/robot_info.json"
        with open(robot_info_path, "r") as f:
            robot_info = json.load(f)

        aihub_gt["robot_info"] = {
            "robot_type": robot_info["robot_type"], # 4.1
            "robot_joint_name": robot_info["robot_joint_name"], # 4.2
            "robot_joint_position": [robot_info["robot_joint_position"][(int(im_id) - 1)]], # 4.3
            "robot_joint_velocity": [robot_info["robot_joint_velocity"][(int(im_id) - 1)]], # 4.4
            "gripper_type": robot_info["gripper_type"], #4.5
            "gripper_joint_name": robot_info["gripper_joint_name"], # 4.6
            "gripper_joint_position": [robot_info["gripper_joint_posision"][(int(im_id) - 1)]], # 4.7
        }

        
        new_file_name = "H3_1_{:06d}_{:06d}".format(scene_id, im_id)
        shutil.copy(scene_path + "/rgb/{:06d}.png".format(im_id), new_rgb_path + "/{}.png".format(new_file_name))
        shutil.copy(scene_path + "/depth/{:06d}.png".format(im_id), new_depth_path + "/{}.png".format(new_file_name))
        shutil.copy(scene_path + "/pcd/{:06d}.pcd".format(im_id), new_pcd_path + "/{}.pcd".format(new_file_name))

        with open(new_gt_path + "/{}.json".format(new_file_name), "w") as f:
            json.dump(aihub_gt, f, indent=4, ensure_ascii=False)







