import cv2
import json
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import argparse

def fill_hole(cnd_target):
    cnd_target = cv2.morphologyEx(cnd_target.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), None, None, 1, cv2.BORDER_REFLECT101)
    return cnd_target


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--is_real', action="store_true")
    parser.add_argument('--n_scenes', type=int, help='number of total scenes to be processed')

    args = parser.parse_args()

    is_real = args.is_real
    n_scenes = args.n_scenes

    home_path = '/home/ailab'
    model_path = f"{home_path}/OccludedObjectDataset/ours/data1/models"
    if is_real:
        dataset_path = f"{home_path}/OccludedObjectDataset/ours/data2/data2_real_source/all"
    else:
        dataset_path = f"{home_path}/OccludedObjectDataset/ours/data2/data2_syn_source/train_pbr"


    # path
    scene_ids = sorted([int(x) for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))])
    new_scene_ids = []
    for scene_id in scene_ids:
        scene_gt_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_gt_{:06d}.json".format(scene_id))
        if not os.path.exists(scene_gt_path):
            continue
        with open(scene_gt_path) as gt_file:
            anno_obj = json.load(gt_file)
        is_all_gt_labeled = True
        for im_id in range(1, 1000):
            if str(im_id) not in anno_obj.keys():
                is_all_gt_labeled = False
                break
        if is_all_gt_labeled:
            new_scene_ids.append(scene_id)
    scene_ids = sorted(new_scene_ids)[:10]

    not_processed_scene_ids = []
    for scene_id in scene_ids:
        scene_path = os.path.join(dataset_path, "{:06d}".format(scene_id))
        occ_mat_path = os.path.join(scene_path, "occ_mat.json")
        if not os.path.exists(occ_mat_path):
            not_processed_scene_ids.append(scene_id)
            continue
        with open(occ_mat_path, 'r') as f:
            file = json.load(f)
        not_processed = []
        for j in range(1, 53):
            if str(j) not in file.keys():
                not_processed_scene_ids.append(scene_id)
                break

        print(len(scene_ids))

