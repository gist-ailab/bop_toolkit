import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--is_real', action="store_true")
    parser.add_argument('--n_scenes', type=int, help='number of total scenes to be processed')

    args = parser.parse_args()

    is_real = args.is_real

    home_path = '/'

    img_id_range = range(0, 1000)
    scene_ids =  list(range(20, 30))
    for scene_id in tqdm(scene_ids):
        print("Process scene {} [from {} to {}]".format(scene_id, scene_ids[0], scene_ids[-1]))
        for im_id in tqdm(img_id_range):
            os.system("/home/seung/anaconda3/envs/bop_toolkit/bin/python aihub/data2/calc_gt_orders.py --scene_id {} --im_id {}".format(scene_id, im_id))