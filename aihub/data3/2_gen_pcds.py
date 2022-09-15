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


dataset_root = "/home/seung/OccludedObjectDataset/ours/data3/data3_1_raw"

camera_names = ["rs_d415", "rs_d435", "azure_kinect"]
bound = [[-1.0, 0.5], [-0.5, 0.5], [-0.05, 1.5]]



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
        depth_img = np.float32(depth_img) / depth_scale / 1000

        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                        cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                                    depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)


        se3 = np.eye(4)
        se3[:3, :3] = cam_R_w2c
        se3[:3, 3] = cam_t_w2c * 0.001
        pcd = pcd.transform(se3)

        min_bound = np.array([bound[0][0], bound[1][0], bound[2][0]])
        max_bound = np.array([bound[0][1], bound[1][1], bound[2][1]])
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

        
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        std = 1.0
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
        pcd = pcd.select_by_index(ind)

        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()
        pcd = pcd.transform(np.linalg.inv(se3))
        o3d.io.write_point_cloud(os.path.join(scene_folder_path, "pcd", i2s(image_number) + ".pcd"), pcd)

