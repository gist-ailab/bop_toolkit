import glob
import json
import os
import numpy as np
import cv2
from pycocotools import mask as m

def mask2rle(im):
    im = np.array(im, order='F', dtype=bool)
    rle = m.encode(im)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle



dataset_path = "/home/seung/OccludedObjectDataset/data2_source"
obj_id2cls_path = "/home/seung/OccludedObjectDataset/data2_source/object_id2class.json"

with open(obj_id2cls_path, 'r') as f:
    obj_id2cls = json.load(f)

scene_ids = [38, 39]

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
            "scene_type": "bin",
            "scene_id": scene_id,
        }
        aihub_gt["object_type"] = []
        
        for obj_gt in scene_gt[str(im_id)]:
            obj_id = str(obj_gt["obj_id"])
            aihub_gt["object_type"].append(
                {
                    "data_id": "{:06d}{:06d}{:06d}".format(scene_id, im_id, obj_gt["inst_id"]),
                    "super_class": obj_id2cls[obj_id]["super_class"],
                    "semantic_class": obj_id2cls[obj_id]["semantic_class"],
                    "instance_class": obj_id2cls[obj_id]["instance_class"],
                }
            )
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


        aihub_gt["object_6d_pose_info"] = []
        aihub_gt["object_mask_info"] = []
        aihub_gt["occlusion_info"] = []
        for idx, obj_gt in enumerate(scene_gt[str(im_id)]):
            object_pose = np.eye(4)
            object_pose[:3, :3] = np.array(obj_gt["cam_R_m2c"]).reshape(3, 3)
            object_pose[:3, 3] = np.array(obj_gt["cam_t_m2c"])
            object_pose = object_pose.tolist()
            model_info = "./models/{:06d}.ply".format(obj_gt["obj_id"])
            aihub_gt["object_6d_pose_info"].append({
                "transformation_matrix": object_pose,
                "model_info": model_info,
            })
            print(im_id, idx)
            amodal_mask = cv2.imread(scene_path + "/mask/{:06d}_{:06d}.png".format(im_id, idx))[:, :, 0]
            visible_mask = cv2.imread(scene_path + "/mask_visib/{:06d}_{:06d}.png".format(im_id, idx))[:, :, 0]
            invisible_mask = cv2.bitwise_xor(amodal_mask, visible_mask)

            aihub_gt["object_mask_info"].append({
                "visible_mask": mask2rle(visible_mask),
                "invisible_mask": mask2rle(invisible_mask),
                "amodal_mask": mask2rle(amodal_mask),
            })

        # occlusion info
        aihub_gt["occlusion_info"] = {}
        # with open(scene_path + "/occ_mat.json", "r") as f:
            # occ_mat = np.array(json.load(f))
        occlusion_order = []
        #!TODO: occlusion_order is not correct
        # if len(scene_gt[str(im_id)]) == len(occ_mat[0]):
            # for idx, obj_gt in enumerate(scene_gt[str(im_id)]):
            #     for k, v in enumerate(occ_mat[idx, :]):
            #         if v == 1:
            #             occlusion_order.append(
            #                 {"order": "{}<{}".format(scene_gt[str(im_id)][idx]["obj_id"], scene_gt[str(im_id)][k]['obj_id']), 
            #                 "overlap": True})
            #     for k, v in enumerate(occ_mat[:, idx]):
            #         if v == 1:
            #             occlusion_order.append(
            #                 {"order": "{}>{}".format(scene_gt[str(im_id)][idx]["obj_id"], scene_gt[str(im_id)][k]['obj_id']), 
            #                 "overlap": True})
            # aihub_gt["occlusion_info"]["occlusion_order"] = occlusion_order


            #!TODO: depth_order is not correct
        depth_order = []
        for i, obj_gt in enumerate(scene_gt[str(im_id)]):
            for j, obj_gt in enumerate(scene_gt[str(im_id)]):
                dist_i = np.linalg.norm(np.array(scene_gt[str(im_id)][i]["cam_t_m2c"]))
                dist_j = np.linalg.norm(np.array(scene_gt[str(im_id)][j]["cam_t_m2c"]))
                if dist_i < dist_j:
                    operator = "<"
                else:
                    operator = ">"
                depth_order.append({"order": "{}{}{}".format(scene_gt[str(im_id)][i]["obj_id"], operator, scene_gt[str(im_id)][j]['obj_id']), "overlap": True})
        aihub_gt["occlusion_info"]["depth_order"] = depth_order
        aihub_gt["occlusion_info"]["occlusion_order"] = depth_order
        # else:
            # print("Error: occ_mat and scene_gt are not matched")

        with open(gt_folder_path + "/{:06d}.json".format(im_id), "w") as f:
            json.dump(aihub_gt, f, indent=4, ensure_ascii=False)


