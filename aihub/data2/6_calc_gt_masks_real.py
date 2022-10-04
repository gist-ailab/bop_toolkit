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
    parser.add_argument('--n_scenes', type=int, help='number of total scenes to be processed')
    parser.add_argument('--n_proc', type=int, help='number of process')
    parser.add_argument('--proc', type=int, help='process id')
    parser.add_argument('--img', type=int, help='image id')

    args = parser.parse_args()

    n_scenes = args.n_scenes
    n_proc = args.n_proc
    proc = args.proc

    home_path = '/home/ailab'
    model_path = f"{home_path}/OccludedObjectDataset/ours/data1/models"

    dataset_path = f"{home_path}/OccludedObjectDataset/ours/data2/data2_real_source/all"
    img_id_range = range(1, 53)

    # path
    scene_ids = sorted([int(x) for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))])
    new_scene_ids = []
    for scene_id in scene_ids[25:]:
        scene_gt_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_gt_{:06d}.json".format(scene_id))
        if not os.path.exists(scene_gt_path):
            # print("Skip scene {} (GT file not found).".format(scene_id))
            continue
        with open(scene_gt_path) as gt_file:
            anno_obj = json.load(gt_file)
        is_all_gt_labeled = True
        for im_id in img_id_range:
            if str(im_id) not in anno_obj.keys():
                print(im_id, scene_id)
                is_all_gt_labeled = False
                break
        if is_all_gt_labeled:
            new_scene_ids.append(scene_id)
    scene_ids = new_scene_ids
    scene_target_i = (n_scenes // n_proc) * (proc - 1)
    scene_target_f = (n_scenes // n_proc) * proc
    scene_ids = scene_ids[:n_scenes][scene_target_i:scene_target_f]
    for scene_id in tqdm(scene_ids):
        print("Process scene {} [from {} to {}]".format(scene_id, scene_ids[0], scene_ids[-1]))
        for im_id in tqdm(img_id_range):
            os.system("/home/ailab/anaconda3/envs/bop_toolkit/bin/python aihub/data2/calc_gt_masks.py --scene_id {} --im_id {}".format(scene_id, im_id))