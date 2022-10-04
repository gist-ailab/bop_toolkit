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


dataset_root = "/home/ailab/OccludedObjectDataset/ours/data3/data3_1_real_source"
aihub_path = "/home/ailab/OccludedObjectDataset/aihub"


scene_id_to_obj_id = {
    "1": "1",
    "2": "11",
    "3": "12",
    "4": "13",
    "5": "14",
    "6": "15",
    "7": "16",
    "8": "17",
    "9": "18",
    "10": "19",
    "11": "21",
    "12": "25",
    "13": "28",
    "14": "31",
    "15": "33",
    "16": "35",
    "17": "38",
    "18": "45",
    "19": "50",
    "20": "51",
    "21": "52",
    "22": "53",
    "23": "54",
    "24": "56",
    "25": "60",
    "26": "61",
    "27": "76",
    "28": "78",
    "29": "80",
    "0": "82"
    }


scene_id_to_object_set = {
    "1": "ycb",
    "2": "ycb",
    "3": "ycb",
    "4": "ycb",
    "5": "ycb",
    "6": "ycb",
    "7": "ycb",
    "8": "ycb",
    "9": "ycb",
    "10": "hope",
    "11": "hope",
    "12": "hope",
    "13": "hope",
    "14": "hope",
    "15": "hope",
    "16": "hope",
    "17": "hope",
    "18": "hope",
    "19": "ycb",
    "20": "ycb",
    "21": "ycb",
    "22": "ycb",
    "23": "ycb",
    "24": "ycb",
    "25": "ycb",
    "26": "ycb",
    "27": "ycb",
    "28": "ycb",
    "29": "ycb",
    "0": "ycb",
}




scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])

for scene_id in tqdm(scene_ids):

    scene_path = os.path.join(dataset_root, "{0:06d}".format(scene_id))
    scene_gt_file = os.path.join(dataset_root, "{0:06d}".format(scene_id), "scene_gt_{0:06d}.json".format(scene_id))

    if not os.path.exists(scene_gt_file):
        print("scene_gt_file not exists: ", scene_gt_file)
        continue
    scene_gt = json.load(open(scene_gt_file, 'r'))
    sub_dir_1, sub_dir_2 =  "UR5", "2-finger(robotiq-2f-140)"
    sub_dir_1_path = os.path.join(aihub_path, '원천데이터', '로봇-물체파지', "실제", sub_dir_1)
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
    if (int(scene_id) - 1) % 120 < 96:
        split = "train"
    elif (int(scene_id) - 1) % 120 < 108:
        split = "val"
    else:
        split = "test"

    for im_id in tqdm(im_ids):
        aihub_gt = {}
        
        # 1. scene_info
        aihub_gt["scene_info"] = {
                "object_set": scene_id_to_object_set[str(scene_id% 30)], # 1.1
                "scene_id": int(scene_id), # 1.2
                "image_id": int(im_id), # 1.3
                "environment": "table",
                "background": "office_partition", # 1.5
                "split": split
            }

        scene_camera_path = scene_path + "/scene_camera.json"
        with open(scene_camera_path, "r") as f:
            scene_camera = json.load(f)

        camera_idx = im_id % 3
        if camera_idx == 1:
            camera_type = "realsense_d415"
            width, height = 1920, 1080
        elif camera_idx == 2:
            camera_type = "realsense_d435"
            width, height = 1920, 1080
        elif camera_idx == 0:
            camera_type = "azure_kinect"
            width, height = 3840, 2160

        # 2. camera_info
        aihub_gt["camera_info"] = { 
            'cam_R_w2c': scene_camera[str(im_id)]["cam_R_w2c"], # 2.1 
            'cam_t_w2c': scene_camera[str(im_id)]["cam_t_w2c"], # 2.2
            "cam_K": scene_camera[str(im_id)]["cam_K"], # 2.3
            'depth_scale': float(scene_camera[str(im_id)]["depth_scale"]), # 2.4
            "resolution": [height, width], # 2.5
            "camera_type": camera_type, # 2.6
            "secs": int(scene_camera[str(im_id)]["secs"]), # 2.7
            "nsecs": int(scene_camera[str(im_id)]["nsecs"]), # 2.8
        }


        # occlusion order
        occlusion_order = []
        # depth order
        depth_order = []
       
        # 3. annotation
        aihub_gt["annotation"] = []
        if str(im_id) in scene_gt:
            for idx, obj_gt in enumerate(scene_gt[str(im_id)]):
                if not os.path.exists(scene_path + "/mask/{:06d}_{:06d}.png".format(im_id, idx)):
                    continue
                amodal_mask = cv2.imread(scene_path + "/mask/{:06d}_{:06d}.png".format(im_id, idx))[:, :, 0]
                visible_mask = cv2.imread(scene_path + "/mask_visib/{:06d}_{:06d}.png".format(im_id, idx))[:, :, 0]
                if np.sum(visible_mask) / np.sum(amodal_mask) > 0.95:
                    invisible_mask = np.zeros_like(amodal_mask)
                else:
                    invisible_mask = cv2.bitwise_xor(amodal_mask, visible_mask)
                target_occlusion_order = [x for x in occlusion_order if int(idx) in [int(x) for x in x["order"].split("&")[0].split("<")]]
                target_depth_order = [x for x in depth_order if int(idx) in [int(x) for x in x["order"].split("<")]]
                if "inst_id" not in obj_gt.keys():
                    obj_gt["inst_id"] = 1
                aihub_gt["annotation"].append({
                    "object_id": obj_gt["obj_id"], # 3.1
                    "instance_id": obj_gt["inst_id"], # 3.2
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
            "robot_joint_position": [robot_info["robot_joint_position"][(int(im_id) - 1) // 3]], # 4.3
            "robot_joint_velocity": [robot_info["robot_joint_velocity"][(int(im_id) - 1) // 3]], # 4.4
            "gripper_type": robot_info["gripper_type"], #4.5
            "gripper_joint_name": robot_info["gripper_joint_name"], # 4.6
            "gripper_joint_position": [robot_info["gripper_joint_position"][(int(im_id) - 1) // 3]], # 4.7
            "command_type": robot_info["command_type"], # 4.8
            "command_value": [robot_info["command_value"][(int(im_id) - 1) // 3]], # 4.9
        }

        
        new_file_name = "H3_2_{:06d}_{:06d}".format(scene_id, im_id)
        shutil.copy(scene_path + "/rgb/{:06d}.png".format(im_id), new_rgb_path + "/{}.png".format(new_file_name))
        shutil.copy(scene_path + "/depth/{:06d}.png".format(im_id), new_depth_path + "/{}.png".format(new_file_name))
        shutil.copy(scene_path + "/pcd/{:06d}.pcd".format(im_id), new_pcd_path + "/{}.pcd".format(new_file_name))

        with open(new_gt_path + "/{}.json".format(new_file_name), "w") as f:
            json.dump(aihub_gt, f, indent=4, ensure_ascii=False)







