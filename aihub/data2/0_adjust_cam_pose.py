import os
import glob
import json
import numpy as np
from tqdm import tqdm
import json
import pandas as pd

ood_root = os.environ['OOD_ROOT']
dataset_root = os.path.join(ood_root, 'ours/data2/data2_real_source/all')

def i2s(num):
    return "{0:06d}".format(int(float(num)))



sch_file = 'assets/scene_info.xlsx'
sch_data = pd.read_excel(sch_file, engine='openpyxl')

calibrated_results = "/home/seung/catkin_ws/src/gail-camera-manager/assets/data2/calibrated_results.json"
world_cam_pose_path = "/home/seung/catkin_ws/src/gail-camera-manager/assets/data2/"

calibrated_results = json.load(open(calibrated_results))

scene_ids = []
envs = []
for date, scene_id, env in zip(sch_data["취득 일자"], sch_data["scene_number"], sch_data["환경"]):
    try:
        int(scene_id)
    except:
        continue

    scene_ids.append(int(scene_id))
    envs.append(env.lower())

for scene_id, env in tqdm(zip(scene_ids, envs)):

    scene_folder_path = os.path.join(dataset_root, i2s(scene_id))
    if not os.path.isdir(scene_folder_path):
        print(scene_folder_path)
        continue
    scene_number = os.path.basename(scene_folder_path)
    scene_camera_info_path = os.path.join(dataset_root, scene_number, "scene_camera.json")
    with open(scene_camera_info_path, 'r') as j_file:
        scene_camera_info = json.load(j_file)
    
    env = env.replace("-", "")
    env = env.replace("_", "")
    if "table" in env:
        env = "table"
    with open(world_cam_pose_path + 'world_cam_poses_{}.json'.format(env), 'r') as f:
        world_cam_poses = json.load(f)

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
    scene_camera_info['0'] = scene_camera_info['1']
    scene_camera_info['-1'] = scene_camera_info['1']
    scene_camera_info['-2'] = scene_camera_info['2']
    scene_camera_info['-3'] = scene_camera_info['3']
    scene_camera_info['-4'] = scene_camera_info['4']
    print("Saving {}".format(scene_camera_info_path))
    with open(scene_camera_info_path, 'w') as j_file:
        json.dump(scene_camera_info, j_file, indent=2)
