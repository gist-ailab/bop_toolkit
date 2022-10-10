import os, sys, glob
import json
import cv2
import numpy as np
from tqdm import tqdm
import copy
import open3d as o3d
import pandas as pd

treg = o3d.t.pipelines.registration 
import open3d as o3d
import numpy as np 
from tqdm import tqdm
import shutil



dates = ["22.10.06"]

sch_file = 'assets/scene_info.xlsx'
sch_data = pd.read_excel(sch_file, engine='openpyxl')

dataset_root = "/OccludedObjectDataset/ours/data2/data2_real_source/all"

camera_names = ["rs_d415", "rs_d435", "azure_kinect", "zivid"]
bin_scene_ids = list(range(1, 101)) + list(range(301, 401))
shelf_scene_ids = list(range(101, 201))
table_scene_ids = list(range(201, 301))
bounds = {
    "bin": [[-1.2, -0.25], [-0.4, 0.4], [-0.615, 0.0]],
    "shelf": [[-0.70, 0.0], [-0.6, 0.6], [-0.5, 1.0]],
    "table": [[-1.5, -0.5], [-0.6, 0.6], [-0.5, 0.5]],
}
num_imgs_per_folder = 52
voxel_size = 0.0025

def i2s(num):
    return "{0:06d}".format(num)


scene_ids = []
envs = []
for date, scene_id, env in zip(sch_data["취득 일자"], sch_data["scene_number"], sch_data["환경"]):
    if date in dates:
        scene_ids.append(scene_id)
        envs.append(env.lower())

for scene_id, env in zip(tqdm(scene_ids), envs):
    if "bin" in env:
        env = "bin"
    elif "shelf" in env:
        env = "shelf"
    elif "table" in env:
        env = "table"
    else:
        print("Unknown env: {}".format(env))
        exit()

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    if not os.path.isdir(scene_folder_path):
        continue
    print("Processing {}".format(scene_folder_path))
    scene_number = os.path.basename(scene_folder_path)
    n_global_image_number = int(num_imgs_per_folder / len(camera_names))

    pcds = {}
    se3s = {}
    for camera_name in camera_names:
        pcds[camera_name] = {}
        se3s[camera_name] = {}
    se3s_base_to_camera_at_center = {}
    scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    with open(scene_camera_info_path, 'r') as j_file:
        scene_camera_info = json.load(j_file)

    bound = bounds[env]

    for global_image_number in tqdm(range(1, n_global_image_number + 1)):

        if not os.path.exists(os.path.join(scene_folder_path, "pcd")):
            os.makedirs(os.path.join(scene_folder_path, "pcd"))

        for idx, camera_name in enumerate(camera_names):
            # read camera_info
            image_number = len(camera_names) * (global_image_number - 1) + idx + 1
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
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)


            se3 = np.eye(4)
            se3[:3, :3] = cam_R_w2c
            se3[:3, 3] = cam_t_w2c * 0.001
            pcd = pcd.transform(se3)
            min_bound = np.array([bound[0][0], bound[1][0], bound[2][0]])
            max_bound = np.array([bound[0][1], bound[1][1], bound[2][1]])
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

            std = 0.5 if scene_id < 100 else 1.0
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
            pcd = pcd.select_by_index(ind)

            if not pcd.has_normals():
                pcd.estimate_normals()
            pcd.normalize_normals()

            pcd = pcd.transform(np.linalg.inv(se3))
            o3d.io.write_point_cloud(os.path.join(scene_folder_path, "pcd", i2s(image_number) + ".pcd"), pcd)


            # to merge all point clouds
            
            if global_image_number == 1:
                se3s_base_to_camera_at_center[camera_name] = copy.deepcopy(se3)
            se3s[camera_name][global_image_number] = se3
            pcd = pcd.transform(se3)
            pcds[camera_name][global_image_number] = pcd
    

    scene_camera_info['0'] = scene_camera_info['1']
    scene_camera_info['-1'] = scene_camera_info['1']
    scene_camera_info['-2'] = scene_camera_info['2']
    scene_camera_info['-3'] = scene_camera_info['3']
    scene_camera_info['-4'] = scene_camera_info['4']
    for scene_camera in scene_camera_info.keys():
        scene_camera_info[scene_camera]['cam_t_w2c'] = np.array(np.array(scene_camera_info[scene_camera]['cam_t_w2c'])).tolist()
    new_scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    
    with open(new_scene_camera_info_path, 'w') as j_file:
        json.dump(scene_camera_info, j_file, indent=2)

    camera_name_to_pcd_number = {
        "rs_d415": "-000001",
        "rs_d435": "-000002",
        "azure_kinect": "-000003",
        "zivid": "-000004",
    }

    camera_name_to_image_number = {
        "rs_d415": "000001",
        "rs_d435": "000002",
        "azure_kinect": "000003",
        "zivid": "000004",
    }

    for i, camera_name in enumerate(pcds.keys()):
        for j, global_image_number in enumerate(pcds[camera_name].keys()):
            target = pcds[camera_name][global_image_number]


            if i == 0 and j == 0:
                pcds_all_cam = target
            if j == 0:
                pcds_this_cam = target
            else:
                pcds_all_cam += target
                pcds_this_cam += target
        
        pcds_this_cam = pcds_this_cam.voxel_down_sample(voxel_size=voxel_size)


        pcds_this_cam = pcds_this_cam.transform(np.linalg.inv(se3s_base_to_camera_at_center[camera_name]))

        o3d.io.write_point_cloud(os.path.join(scene_folder_path, "pcd", camera_name_to_pcd_number[camera_name] + ".pcd"), pcds_this_cam)
        shutil.copy(os.path.join(scene_folder_path, "rgb", camera_name_to_image_number[camera_name] + ".png"),  os.path.join(scene_folder_path, "rgb", camera_name_to_pcd_number[camera_name] + ".png"))
        shutil.copy(os.path.join(scene_folder_path, "depth", camera_name_to_image_number[camera_name] + ".png"),  os.path.join(scene_folder_path, "depth", camera_name_to_pcd_number[camera_name] + ".png"))

    pcds_all_cam = pcds_all_cam.voxel_down_sample(voxel_size=voxel_size*1.0)
    pcds_all_cam = pcds_all_cam.transform(np.linalg.inv(se3s_base_to_camera_at_center['rs_d415']))
    # o3d.visualization.draw_geometries([pcds_this_cam])

    o3d.io.write_point_cloud(os.path.join(scene_folder_path, "pcd", "000000.pcd"), pcds_all_cam)
    shutil.copy(os.path.join(scene_folder_path, "rgb", "000001.png"),  os.path.join(scene_folder_path, "rgb", "000000.png"))
    shutil.copy(os.path.join(scene_folder_path, "depth", "000001.png"),  os.path.join(scene_folder_path, "depth", "000000.png"))
