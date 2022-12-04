import os, sys, glob
import json
import cv2
import numpy as np
from tqdm import tqdm
import copy
import open3d as o3d
import pandas as pd

import open3d as o3d
import numpy as np 
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import imageio
ood_root = os.environ['OOD_ROOT']
dataset_root = os.path.join(ood_root, 'ours/data3/data3_syn_source/')


def depth_image_from_distance_image(distance, intrinsics):
  """Computes depth image from distance image.
  
  Background pixels have depth of 0
  
  Args:
      distance: HxW float array (meters)
      intrinsics: 3x3 float array
  
  Returns:
      z: HxW float array (meters)
  
  """
  fx = intrinsics[0, 0]
  cx = intrinsics[0, 2]
  fy = intrinsics[1, 1]
  cy = intrinsics[1, 2]
  
  height, width = distance.shape
  xlin = np.linspace(0, width - 1, width)
  ylin = np.linspace(0, height - 1, height)
  px, py = np.meshgrid(xlin, ylin)
  
  x_over_z = (px - cx) / fx
  y_over_z = (py - cy) / fy
  
  z = distance / np.sqrt(1. + x_over_z**2 + y_over_z**2)
  return z

scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
for scene_id in tqdm(scene_ids): 

    if int(scene_id) < 89:
        continue

    img_ids = list(range(1, 251))
    scene_camera_info_path = os.path.join(dataset_root, "{0:06d}".format(scene_id), "scene_camera.json")


    for img_id in img_ids:
        print("scene_id: ", scene_id, "img_id: ", img_id)
        with open(scene_camera_info_path, 'r') as j_file:
            scene_camera_info = json.load(j_file)
        img_path = os.path.join(dataset_root, "{0:06d}".format(scene_id), 'depth', '{0:06d}.png'.format(img_id))
        new_img_path = os.path.join(dataset_root, "{0:06d}".format(scene_id), 'depth_adjusted', '{0:06d}.png'.format(img_id))
        if not os.path.exists(os.path.join(dataset_root, "{0:06d}".format(scene_id), 'depth_adjusted')):
            os.makedirs(os.path.join(dataset_root, "{0:06d}".format(scene_id), 'depth_adjusted'))
        depth = cv2.imread(img_path, -1)
        cam_info = scene_camera_info[str(img_id)]
        intrinsics = np.array(cam_info['cam_K']).reshape(3, 3)
        depth = np.uint16(depth_image_from_distance_image(depth, intrinsics) )
        cv2.imwrite(new_img_path, depth)
        