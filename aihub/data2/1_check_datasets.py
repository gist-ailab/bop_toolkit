import glob
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

dataset_root = "/home/seung/OccludedObjectDataset/data2_source"
num_imgs_per_folder = 52

# 1. merge all folders into a single folder, then check whether some images are missing
for scene_folder_path in tqdm(glob.glob(os.path.join(dataset_root) + "/*")):
    if not os.path.isdir(scene_folder_path):
        continue
    # print("checking scene folder: ", scene_folder_path)
    assert int(os.path.basename(scene_folder_path)), "Invalid scene folder name: {}".format(scene_folder_path)

    # check whether some images are missing
    rgb_paths = glob.glob(os.path.join(scene_folder_path) + "/rgb/*.png")
    depth_paths = glob.glob(os.path.join(scene_folder_path) + "/depth/*.png")

    if len(rgb_paths) not in [num_imgs_per_folder, 57]:
        print("Invalid number of rgb images: {}".format(scene_folder_path))
        continue

    assert len(rgb_paths) == len(depth_paths), "Number of rgb and depth images are not equal: {} vs {}".format(len(rgb_paths), len(depth_paths))

    # # check whether some images are broken
    # for rgb_path, depth_path in zip(rgb_paths, depth_paths):
    #     assert cv2.imread(rgb_path).any() is not None
    #     assert cv2.imread(depth_path).any() is not None




