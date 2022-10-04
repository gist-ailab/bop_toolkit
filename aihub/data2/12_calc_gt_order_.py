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
    parser.add_argument('--n_proc', type=int, help='number of process')
    parser.add_argument('--proc', type=int, help='process id')

    args = parser.parse_args()

    is_real = args.is_real
    n_scenes = args.n_scenes
    n_proc = args.n_proc
    proc = args.proc

    home_path = '/'
    model_path = f"{home_path}/OccludedObjectDataset/ours/data1/models"

    dataset_path = f"{home_path}/OccludedObjectDataset/ours/data2/data2_real_source/all"
    img_id_range = range(1, 53)


    scene_ids =  [35, 36, 37, 38, 39, 40, 41, 73, 74, 75, 76, 77, 78, 79, 80]
    # scene_ids = [ 81, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 202, 203, 204]
    for scene_id in tqdm(scene_ids):
        print("Process scene {} [from {} to {}]".format(scene_id, scene_ids[0], scene_ids[-1]))
        for im_id in tqdm(img_id_range):
            os.system("/home/seung/anaconda3/envs/bop_toolkit/bin/python aihub/data2/calc_gt_orders.py --is_real --scene_id {} --im_id {}".format(scene_id, im_id))