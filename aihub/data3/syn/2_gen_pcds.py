import os, sys, glob
import shelve
import json
import cv2
import numpy as np
import imageio
from tqdm import tqdm
import copy
import open3d as o3d
import yaml
import time

treg = o3d.t.pipelines.registration 
import open3d as o3d
import numpy as np 
from scipy.spatial import cKDTree
from tqdm import tqdm
import shutil


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

def i2s(num):
    return "{0:06d}".format(num)


scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
for scene_id in tqdm(scene_ids):

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    if not os.path.isdir(scene_folder_path):
        continue
    print("Processing {}".format(scene_folder_path))
    scene_number = os.path.basename(scene_folder_path)
    scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    with open(scene_camera_info_path, 'r') as j_file:
        scene_camera_info = json.load(j_file)

    voxel_size = 0.0025
    im_ids = sorted([int(x.split(".")[0]) for x in os.listdir(os.path.join(dataset_root, scene_number, "rgb"))])

    if not os.path.exists(os.path.join(scene_folder_path, "pcd")):
        os.makedirs(os.path.join(scene_folder_path, "pcd"))
    
    for image_number in tqdm(im_ids):

        if image_number > 250:
            break

        # read camera_info
        cam_K = np.array(scene_camera_info[str(image_number)]["cam_K"]).reshape(3, 3)
        depth_scale = scene_camera_info[str(image_number)]["depth_scale"]
        cam_R_w2c = np.array(scene_camera_info[str(image_number)]["cam_R_w2c"]).reshape(3,3)
        cam_t_w2c = np.array(scene_camera_info[str(image_number)]["cam_t_w2c"]) 
        

        # read rgb, depth
        rgb_path = os.path.join(dataset_root, scene_number, "rgb", i2s(image_number) + ".png")
        depth_path = os.path.join(dataset_root, scene_number, "depth", i2s(image_number) + ".png")
        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, -1)
        depth_img = depth_image_from_distance_image(depth_img, cam_K)
        depth_img = np.float32(depth_img) / depth_scale / 1000

        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        height, width = depth_img.shape[:2]
        fx, fy, cx, cy = cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                                    depth_scale=1, depth_trunc=10.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=False)


        se3 = np.eye(4)
        se3[:3, :3] = cam_R_w2c
        se3[:3, 3] = cam_t_w2c * 0.001
        pcd = pcd.transform(se3)

        
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()
        pcd = pcd.transform(np.linalg.inv(se3))
        o3d.io.write_point_cloud(os.path.join(scene_folder_path, "pcd", i2s(image_number) + ".pcd"), pcd)
        # o3d.visualization.draw_geometries([pcd])

