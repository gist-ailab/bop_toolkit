import os
import glob
import json
import numpy as np
from tqdm import tqdm
import json

dataset_root = "/home/seung/OccludedObjectDataset/ours/data2/data2_real_source/all"


def i2s(num):
    return "{0:06d}".format(num)


scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
scene_ids = [x for x in scene_ids if x > 300]

scene_ids = [384, 385, 386, 387, 388, 390, ]

calibrated_results = "/home/seung/catkin_ws/src/gail-camera-manager/assets/data2/calibrated_results.json"
world_cam_poses = "/home/seung/catkin_ws/src/gail-camera-manager/assets/data2/world_cam_poses_table.json"
with open(world_cam_poses, 'r') as f:
    world_cam_poses = json.load(f)
calibrated_results = json.load(open(calibrated_results))


for scene_id in tqdm(scene_ids):

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    if not os.path.isdir(scene_folder_path):
        continue
    print("Processing {}".format(scene_folder_path))
    scene_number = os.path.basename(scene_folder_path)
    scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    with open(scene_camera_info_path, 'r') as j_file:
        scene_camera_info = json.load(j_file)

    for image_number in range(1, 53):
        image_number = str(image_number)
        if int(image_number) % 4 == 0:
            camera_name = "zivid"
        elif int(image_number) % 4 == 1:
            camera_name = "rs_d415"
        elif int(image_number) % 4 == 2:
            camera_name = "rs_d435"
        elif int(image_number) % 4 == 3:
            camera_name = "azure_kinect"
        cam_pose_id = ( int(image_number) - 1) // 4 + 1
        if cam_pose_id == 0:
            cam_pose_id = 13
        H_base_to_cam_1 = np.array(world_cam_poses[str(cam_pose_id)]).reshape(4, 4)
        H_cam_1_to_k = np.array(calibrated_results[camera_name]['H']).reshape(4, 4)
        H_base_to_cam = np.matmul(H_base_to_cam_1, H_cam_1_to_k)
        cam_R_w2c = H_base_to_cam[:3, :3].reshape(-1).tolist()
        cam_t_w2c = H_base_to_cam[:3, 3].tolist()

        scene_camera_info[image_number]["cam_R_w2c"] = cam_R_w2c
        scene_camera_info[image_number]["cam_t_w2c"] = cam_t_w2c
    
    with open(scene_camera_info_path, 'w') as j_file:
        json.dump(scene_camera_info, j_file, indent=2)
