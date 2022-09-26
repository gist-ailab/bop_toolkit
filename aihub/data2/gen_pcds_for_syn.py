from ast import Break
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






dataset_root = "/home/seung/OccludedObjectDataset/ours/data2_syn"

camera_names = ["rs_d415", "rs_d435", "azure_kinect", "zivid"]
bin_scene_ids = list(range(1, 101))
shelf_scene_ids = list(range(101, 201))
bounds = {
    "bin": [[-1.2, -0.25], [-0.4, 0.4], [-0.615, 0.0]],
    "shelf": [[-0.70, 0.0], [-0.6, 0.6], [-0.5, 1.0]]
}
num_imgs_per_folder = 1000

def i2s(num):
    return "{0:06d}".format(num)


scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
# scene_ids = [x for x in scene_ids if x in bin_scene_ids][34:]
# scene_ids = [x for x in scene_ids if x in shelf_scene_ids]
# scene_ids = [3, 4, 13, 22, 29, 34, 35, 41, 42, 43, 44, 45, 46, 47, 48, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66, 67, 68, 73, 74, 75, 76, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
scene_ids = [1]

for scene_id in tqdm(scene_ids):

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    if not os.path.isdir(scene_folder_path):
        continue
    print("Processing {}".format(scene_folder_path))
    scene_number = os.path.basename(scene_folder_path)
    n_global_image_number = int(num_imgs_per_folder / len(camera_names))

    pcds = {}
    se3s = {}

    se3s_base_to_camera_at_center = {}
    scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    with open(scene_camera_info_path, 'r') as j_file:
        scene_camera_info = json.load(j_file)

  
    voxel_size = 0.0025

    for image_number in tqdm(range(0, num_imgs_per_folder)):

        if not os.path.exists(os.path.join(scene_folder_path, "pcd")):
            os.makedirs(os.path.join(scene_folder_path, "pcd"))

        # if global_image_number > 5: 
            # continue
            
        cam_K = np.array(scene_camera_info[str(image_number)]["cam_K"]).reshape(3, 3)
        depth_scale = scene_camera_info[str(image_number)]["depth_scale"]
        cam_R_w2c = np.array(scene_camera_info[str(image_number)]["cam_R_w2c"]).reshape(3,3)
        cam_t_w2c = np.array(scene_camera_info[str(image_number)]["cam_t_w2c"]) 
        

        # read rgb, depth
        rgb_path = os.path.join(dataset_root, scene_number, "rgb", i2s(image_number) + ".jpg")
        depth_path = os.path.join(dataset_root, scene_number, "depth", i2s(image_number) + ".png")
        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, -1)
        #!TODO: check depth scale (* -> / )?
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

        se3 = np.eye(4)
        se3[:3, :3] = cam_R_w2c
        se3[:3, 3] = cam_t_w2c * 0.001
        o3d.io.write_point_cloud(os.path.join(scene_folder_path, "pcd", i2s(image_number) + ".pcd"), pcd)

