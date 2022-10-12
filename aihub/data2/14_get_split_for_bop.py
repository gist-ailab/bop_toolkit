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


is_real = True
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
scene_ids = [x for x in scene_ids if int(x) < 205 and int(x) not in [25, 176, 200, 201]]

split_info = {}
object_set_info = {}

for scene_id in tqdm(scene_ids):

    split_info[scene_id] = scene_id_to_scene_info[scene_id]["split"]
    object_set_info[scene_id] = scene_id_to_scene_info[scene_id]["object_set"]


with open("split_info_real.json", "w") as f:
    json.dump(split_info, f, indent=4, ensure_ascii=False)


with open("object_set_info.json", "w") as f:
    json.dump(object_set_info, f, indent=4, ensure_ascii=False)
