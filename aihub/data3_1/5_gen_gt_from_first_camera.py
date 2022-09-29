import cv2
import json
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import copy


if __name__ == "__main__":

    dataset_path = "/home/seung/OccludedObjectDataset/ours/data3/data3_1_real_source"
    model_path = "/home/seung/OccludedObjectDataset/ours/data3/models"
    # path
    scene_ids = sorted([int(x) for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))])
    
    # scene_ids = [x for x in scene_ids if 154 < x < 201]

    for scene_id in tqdm(scene_ids):

        print("Process scene {}".format(scene_id))
        # load scene_gt
        scene_gt_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_gt_{:06d}.json".format(scene_id))
        if not os.path.exists(scene_gt_path):
            print("Skip scene {} (GT file not found).".format(scene_id))
            continue
        with open(scene_gt_path) as scene_gt_file:
            scene_gt = json.load(scene_gt_file)

        # load scene_camera_info
        scene_camera_info_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_camera.json")
        with open(scene_camera_info_path) as scene_camera_info_file:
            scene_camera_info = json.load(scene_camera_info_file)

        # load keyframes
        key_frame_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "keyframes.json")
        with open(key_frame_path) as key_frame_file:
            key_frames = json.load(key_frame_file)
        
        for key_frame in tqdm(key_frames):
            
            im_id = int(key_frame)
            is_main_cam = True if im_id % 3 == 1 else False
            if is_main_cam:
                source_scene_gt = copy.deepcopy(scene_gt[str(im_id)])
                se3_base_to_source = np.eye(4)
                se3_base_to_source[:3, :3] = np.array(scene_camera_info[str(im_id)]["cam_R_w2c"]).reshape(3, 3)
                se3_base_to_source[:3, 3] = np.array(scene_camera_info[str(im_id)]["cam_t_w2c"])
            else:
                se3_base_to_target = np.eye(4)
                se3_base_to_target[:3, :3] = np.array(scene_camera_info[str(im_id)]["cam_R_w2c"]).reshape(3, 3)
                se3_base_to_target[:3, 3] = np.array(scene_camera_info[str(im_id)]["cam_t_w2c"])
                se3_target_to_source = np.matmul(np.linalg.inv(se3_base_to_target), copy.deepcopy(se3_base_to_source))

                target_scene_gt = list()
                for source in source_scene_gt:
                    se3_source_to_object = np.eye(4)
                    se3_source_to_object[:3, :3] = copy.deepcopy(np.array(source['cam_R_m2c']).reshape(3, 3))
                    se3_source_to_object[:3, 3] = copy.deepcopy(np.array(source['cam_t_m2c']))
                    se3_target_to_object = np.matmul(se3_target_to_source, copy.deepcopy(se3_source_to_object))
                    target = copy.deepcopy(source)
                    target['cam_R_m2c'] = se3_target_to_object[:3, :3].reshape(9).tolist() 
                    target['cam_t_m2c'] = se3_target_to_object[:3, 3].tolist()
                    target_scene_gt.append(target)
                scene_gt[str(int(im_id))] = target_scene_gt
        
        with open(scene_gt_path, 'w') as scene_gt_file:
            json.dump(scene_gt, scene_gt_file, indent=4)

            



