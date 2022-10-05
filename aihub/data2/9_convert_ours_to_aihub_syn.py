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


is_real = False
scene_info_path = 'assets/scene_info.xlsx'
if is_real:
    dataset_root = "/home/ailab/OccludedObjectDataset/ours/data2/data2_real_source/all"
else:
    dataset_root = "/home/ailab/OccludedObjectDataset/ours/data2/data2_syn_source/train_pbr"
aihub_path = "/home/ailab/OccludedObjectDataset/aihub"

env_to_background = {
    "bin":
        {
            "1": "office_floor",
            "2": "wood_plate",
            "3": "carpet_floor",
            "4": "wood_tile",
            "5": "cement_tile",
        },
    "shelf":
        {
            "1": "office_partition",
            "2": "red_brick_wall",
            "3": "white_brick_wall",
            "4": "white_wall",
            "5": "cement_wall"
        },
    "table":
            {
            "1": "office_partition",
            "2": "red_brick_wall",
            "3": "white_brick_wall",
            "4": "white_wall",
            "5": "cement_wall"
        },
}

object_set_to_subdir = {
    'public_all': ['혼합', 'PublicObject_혼합'],
    'ycb_all': ['YCB', 'YCB_전체'],
    'ycb_kitchen': ['YCB', 'YCB_주방'],
    'ycb_food': ['YCB', 'YCB_음식'],
    'ycb_tool': ['YCB', 'YCB_도구'], 
    'hope_all': ['HOPE', 'HOPE_전체'],
    'apc_all': ['APC', 'APC_전체'],
}


scene_info = pd.read_excel(scene_info_path, engine='openpyxl')


# get scene info
if is_real:
    scene_ids = scene_info['scene_number']
    scene_types = scene_info['scene_type']
    environments = scene_info['환경']
    backgrounds = scene_info['배경']

    scene_id_to_scene_info = {}
    for scene_id, scene_type, environment, background in zip(scene_ids, scene_types, environments, backgrounds):
        scene_type = scene_type.lower().replace("-", "_")
        object_set = scene_type.split("_")[:2]
        object_set = "_".join(object_set)
        object_set = object_set.replace("object_all", "public_all")
        split = scene_type.split("_")[-1]
        background = background[0] if isinstance(background, str) else background
        scene_id_to_scene_info[int(scene_id)] = {
            "object_set": object_set, # 1.1 
            "scene_id": int(scene_id), # 1.2
            "environment": environment.lower().replace("-", ""), # 1.4
            "background": env_to_background[environment.split('-')[0].lower()][str(background)], # 1.5
            "split": split, # 1.6
        }


scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
scene_ids = scene_ids[1:2]

for scene_id in tqdm(scene_ids):

    scene_path = os.path.join(dataset_root, "{0:06d}".format(scene_id))
    if is_real:
        scene_gt_file = os.path.join(dataset_root, "{0:06d}".format(scene_id), "scene_gt_{0:06d}.json".format(scene_id))
    else:
        scene_gt_file = os.path.join(dataset_root, "{0:06d}".format(scene_id), "scene_gt.json")
    if not os.path.exists(scene_gt_file):
        print("scene_gt_file not exists: ", scene_gt_file)
        continue
    scene_gt = json.load(open(scene_gt_file, 'r'))
    if not is_real:
        object_set = "ycb_all"
    else:
        object_set = scene_id_to_scene_info[int(scene_id)]['object_set']
    sub_dir_1, sub_dir_2 =  object_set_to_subdir[object_set]
    dir_name = "실제" if is_real else "가상"
    sub_dir_1_path = os.path.join(aihub_path, '원천데이터', '다수물체가림', dir_name, sub_dir_1)
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




    # occlusion & depth order info
    with open(scene_path + "/occ_mat.json", "r") as f:
        occ_mats = json.load(f)
    with open(scene_path + "/depth_mat.json", "r") as f:
        depth_mats = json.load(f)
    with open(scene_path + "/is_overlap_matrix.json", "r") as f:
        is_overlap_mats = json.load(f)
    


    for im_id in tqdm(range(853, 1000)):
        aihub_gt = {}
        # 1. scene_info
        if is_real:
            aihub_gt["scene_info"] = {
                    "object_set": scene_id_to_scene_info[int(scene_id)]["object_set"], # 1.1 
                    "scene_id": scene_id_to_scene_info[int(scene_id)]["scene_id"], # 1.2
                    "image_id": int(im_id), # 1.3
                    "environment": scene_id_to_scene_info[int(scene_id)]["environment"], # 1.4
                    "background": scene_id_to_scene_info[int(scene_id)]["background"], # 1.5
                    "split": scene_id_to_scene_info[int(scene_id)]["split"], # 1.6
                }
        else:
            background_info_path = os.path.join(dataset_root, "{0:06d}".format(scene_id), "background.json")
            with open(background_info_path, "r") as f:
                background_info = json.load(f)
            if scene_id % 10 == 9:
                split = "test" 
            elif scene_id % 10 == 8:
                split = "val"
            else:
                split = "train"
            aihub_gt["scene_info"] = {
                    "object_set": "ycb_all", # 1.1 
                    "scene_id": int(scene_id), # 1.2
                    "image_id": int(im_id), # 1.3
                    "environment": "floor", # 1.4
                    "background": background_info[str(scene_id)], # 1.5
                    "split": split, # 1.6
                }

        scene_camera_path = scene_path + "/scene_camera.json"
        with open(scene_camera_path, "r") as f:
            scene_camera = json.load(f)

        camera_idx = im_id % 4
        if camera_idx == 1:
            camera_type = "realsense_d415"
            width, height = 1920, 1080
        elif camera_idx == 2:
            camera_type = "realsense_d435"
            width, height = 1920, 1080
        elif camera_idx == 3:
            camera_type = "azure_kinect"
            width, height = 3840, 2160
        elif camera_idx == 0:
            camera_type = "zivid"
            width, height = 1920, 1200

        # 2. camera_info
        aihub_gt["camera_info"] = { 
            'cam_R_w2c': scene_camera[str(im_id)]["cam_R_w2c"], # 2.1 
            'cam_t_w2c': scene_camera[str(im_id)]["cam_t_w2c"], # 2.2
            "cam_K": scene_camera[str(im_id)]["cam_K"], # 2.3
            'depth_scale': float(scene_camera[str(im_id)]["depth_scale"]), # 2.4
            "resolution": [height, width], # 2.5
            "camera_type": camera_type, # 2.6
        }


        # occlusion order
        anno_ids = list(range(len(scene_gt[str(im_id)])))
        occlusion_order = []
        #!TODO: occlusion_order is not correct
        occ_mat = occ_mats[str(im_id)]
        occ_mat = np.array(occ_mat).reshape(len(anno_ids), len(anno_ids))

        bidir_pairs = []
        for idx_A in range(len(anno_ids)):
            for idx_B in range(len(anno_ids)):
                if idx_A == idx_B:
                    continue
                elif occ_mat[idx_A, idx_B] == 1 and occ_mat[idx_B, idx_A] == 1:
                    if (idx_A, idx_B) not in bidir_pairs and (idx_B, idx_A) not in bidir_pairs:
                        occlusion_order.append(
                            {"order": "{}<{} & {}<{}".format(
                                anno_ids[idx_A], anno_ids[idx_B], anno_ids[idx_B], anno_ids[idx_A])}
                        )
                        bidir_pairs.append((idx_A, idx_B))
                elif occ_mat[idx_A, idx_B] == 1:
                    occlusion_order.append(
                        {"order": "{}<{}".format(
                            anno_ids[idx_A], anno_ids[idx_B])})
                  
        # depth order
        depth_order = []
        depth_mat = depth_mats[str(im_id)]
        depth_mat = np.array(depth_mat).reshape(len(anno_ids), len(anno_ids))
        is_overlap_mat = is_overlap_mats[str(im_id)]
        is_overlap_mat = np.array(is_overlap_mat).reshape(len(anno_ids), len(anno_ids))
        anno_ids = list(range(len(scene_gt[str(im_id)])))
        for idx_A in range(len(anno_ids)):
            for idx_B in range(len(anno_ids)):
                if idx_A == idx_B:
                    continue
                if depth_mat[idx_A, idx_B] == 1:
                    depth_order.append(
                        {"order": "{}<{}".format(
                            anno_ids[idx_A], anno_ids[idx_B]), "overlap": str(is_overlap_mat[idx_A, idx_B] == 1)})

        # 3. annotation
        aihub_gt["annotation"] = []
        for idx, obj_gt in enumerate(scene_gt[str(im_id)]):
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

        
        real_or_syn = 2 if is_real else 1
        new_file_name = "H2_{}_{:06d}_{:06d}".format(real_or_syn, scene_id, im_id)
        if is_real:
            shutil.copy(scene_path + "/rgb/{:06d}.png".format(im_id), new_rgb_path + "/{}.png".format(new_file_name))
        else:
            shutil.copy(scene_path + "/rgb/{:06d}.jpg".format(im_id), new_rgb_path + "/{}.jpg".format(new_file_name))
        shutil.copy(scene_path + "/depth/{:06d}.png".format(im_id), new_depth_path + "/{}.png".format(new_file_name))
        shutil.copy(scene_path + "/pcd/{:06d}.pcd".format(im_id), new_pcd_path + "/{}.pcd".format(new_file_name))
        with open(new_gt_path + "/{}.json".format(new_file_name), "w") as f:
            json.dump(aihub_gt, f, indent=4, ensure_ascii=False)







