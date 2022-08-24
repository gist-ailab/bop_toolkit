import glob
import json
import os
import numpy as np

dataset_path = "/home/seung/OccludedObjectDataset/data2_source"

scene_ids = [2, 11, 15, 38, 39]
for scene_id in scene_ids:
    scene_path = os.path.join(dataset_path, "{:06d}".format(scene_id))
    scene_gt_path = scene_path + "/scene_gt_{:06d}.json".format(scene_id)
    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)
    scene_camera_path = scene_path + "/scene_camera.json"
    with open(scene_camera_path, "r") as f:
        scene_camera = json.load(f)

    gt_folder_path = scene_path + "/gt"
    if not os.path.exists(gt_folder_path):
        os.makedirs(gt_folder_path)

    for im_id in range(1, 53):
        aihub_gt = {}
        aihub_gt["scene_info"] = {
            "scene_type": "table_top",
            "scene_id": scene_id,
        }
        aihub_gt["object_type"] = {
            "data_id": "{:06d}{:06d}".format(scene_id, im_id),
            "super_class": "5",
            "semantic_class": "11",
            "instance_class": "0",
        }
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = np.array(scene_camera[str(im_id)]["cam_R_w2c"]).reshape(3, 3)
        camera_pose[:3, 3] = np.array(scene_camera[str(im_id)]["cam_t_w2c"])
        
        camera_pose = camera_pose.tolist()

        camera_idx = im_id % 4
        if camera_idx == 1:
            camera_type = "realsense_d415"
            width, height = 1920, 1080
        elif camera_idx == 2:
            camera_type = "realsense_d435"
            width, height = 1920, 1080
        elif camera_idx == 3:
            camera_type = "azure_kinect"
            width, height = 3840, 2160
        elif camera_idx == 0:
            camera_type = "zivid"
            width, height = 1920, 1200

        aihub_gt["camera_info"] = {
            "seq": im_id,
            "camera_pose": camera_pose,
            "camera_intrinsic": scene_camera[str(im_id)]["cam_K"],
            "camera_resolution": [height, width],
            "camera_fps": 1,
            "camera_type": camera_type,
        }
        with open(gt_folder_path + "/{:06d}.json".format(im_id), "w") as f:
            json.dump(aihub_gt, f, indent=4)


