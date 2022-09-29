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
# import teaserpp_python
from tqdm import tqdm
import shutil






dataset_root = "/home/seung/OccludedObjectDataset/ours/data2/data2_syn_source/train_pbr"

camera_names = ["rs_d415", "rs_d435", "azure_kinect", "zivid"]
num_imgs_per_folder = 52
voxel_size = 0.0025

def i2s(num):
    return "{0:06d}".format(num)


scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])

for scene_id in tqdm(scene_ids):

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    if not os.path.isdir(scene_folder_path):
        continue
    print("Processing {}".format(scene_folder_path))
    scene_number = os.path.basename(scene_folder_path)

    pcds = {}
    se3s = {}
    for camera_name in camera_names:
        pcds[camera_name] = {}
        se3s[camera_name] = {}
    se3s_base_to_camera_at_center = {}
    scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    with open(scene_camera_info_path, 'r') as j_file:
        scene_camera_info = json.load(j_file)



    if not os.path.exists(os.path.join(scene_folder_path, "pcd")):
        os.makedirs(os.path.join(scene_folder_path, "pcd"))

    for idx, image_number in enumerate(range(0, 1000)):
        # read camera_info
        cam_K = np.array(scene_camera_info[str(image_number)]["cam_K"]).reshape(3, 3)
        depth_scale = scene_camera_info[str(image_number)]["depth_scale"]
        cam_R_w2c = np.array(scene_camera_info[str(image_number)]["cam_R_w2c"]).reshape(3,3)
        cam_t_w2c = np.array(scene_camera_info[str(image_number)]["cam_t_w2c"]) 
        

        # read rgb, depth
        rgb_path = os.path.join(dataset_root, scene_number, "rgb", i2s(image_number) + ".jpg")
        depth_path = os.path.join(dataset_root, scene_number, "depth", i2s(image_number) + ".png")
        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, -1)
        depth_img = np.float32(depth_img) * depth_scale / 1000

        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                        cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                                    depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)


        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()

        o3d.io.write_point_cloud(os.path.join(scene_folder_path, "pcd", i2s(image_number) + ".pcd"), pcd)

