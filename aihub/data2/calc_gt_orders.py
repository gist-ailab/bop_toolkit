import cv2
import json
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import argparse

def fill_hole(cnd_target):
    cnd_target = cv2.morphologyEx(cnd_target.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), None, None, 1, cv2.BORDER_REFLECT101)
    # cnd_target = cv2.morphologyEx(cnd_target.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8), None, None, 1, cv2.BORDER_REFLECT101)
    return cnd_target


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--scene_id', type=int, help='scene id')
    parser.add_argument('--im_id', type=int, help='image id')
    parser.add_argument('--is_real', action="store_true")


    args = parser.parse_args()

    scene_id = args.scene_id
    im_id = args.im_id
    is_real = args.is_real

    home_path = '/home/ailab'
    model_path = f"{home_path}/OccludedObjectDataset/ours/data1/models"

    if is_real:
        dataset_path = f"{home_path}/OccludedObjectDataset/ours/data2/data2_real_source/all"
        img_id_range = range(1, 53)
    else:
        dataset_path = f"{home_path}/OccludedObjectDataset/ours/data2/data2_syn_source/train_pbr"
        img_id_range = range(0, 1000)

    if is_real:
                scene_gt_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_gt_{:06d}.json".format(scene_id))
    else:
        scene_gt_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_gt.json")
    print("Process scene {} image {} \n".format(scene_id, im_id))
    root_data = os.path.join(dataset_path, "{:06d}".format(scene_id))
    if is_real:
        path_rgb = os.path.join(root_data, 'rgb', '{:06d}.png')
    else:
        path_rgb = os.path.join(root_data, 'rgb', '{:06d}.jpg')
    path_depth = os.path.join(root_data, 'depth', '{:06d}.png')
    path_anno_cam = os.path.join(root_data, 'scene_camera.json')
    path_amodal_mask = os.path.join(root_data, 'mask')
    path_visible_mask = os.path.join(root_data, 'mask_visib')
    os.makedirs(path_amodal_mask, exist_ok=True)
    os.makedirs(path_visible_mask, exist_ok=True)

    # load camera pose annotation
    with open(path_anno_cam.format()) as gt_file:
        anno_cam = json.load(gt_file)
    anno_cam = anno_cam[str(im_id)] # type: dict

    # load object pose annotation
    with open(scene_gt_path) as gt_file:
        anno_obj = json.load(gt_file)
    anno_obj = anno_obj[str(im_id)] # type: list

    # load object pointcloud and transform as annotation
    obj_geometries = {}
    obj_depths = {}
    for i, obj in enumerate(anno_obj):
        translation = np.array(np.array(obj['cam_t_m2c']), dtype=np.float64) / 1000  # convert to meter
        orientation = np.array(np.array(obj['cam_R_m2c']), dtype=np.float64)
        transform = np.concatenate((orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1)
        transform_cam_to_obj = np.concatenate(
            (transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform
        # load pointcloud (.ply)
        obj_geometry = o3d.io.read_point_cloud(
            os.path.join(model_path, 'obj_' + f"{int(obj['obj_id']):06}" + '.ply'))
        obj_geometry.points = o3d.utility.Vector3dVector(
            np.array(obj_geometry.points) / 1000)  # convert mm to meter
        # move object
        obj_geometry.transform(transform_cam_to_obj)
        # save in dictionary
        obj_geometries[i] = obj_geometry
        obj_depths[i] = [obj_geometry.get_min_bound()[2], obj_geometry.get_max_bound()[2]]

    # generate offscreen renderer
    rgb_img = cv2.imread(path_rgb.format(im_id))
    img_h, img_w, img_c = rgb_img.shape
    render = o3d.visualization.rendering.OffscreenRenderer(
                                        width=img_w, height=img_h)
    # black background color
    render.scene.set_background([0, 0, 0, 1])
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, [0,0,0])
    

    obj_mtl = o3d.visualization.rendering.MaterialRecord()
    obj_mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    obj_mtl.shader = "defaultUnlit"
    obj_mtl.point_size = 10.0

    # initialize occlusion & depth order (n x n matrix)
    occ_mat = np.zeros((len(obj_geometries), len(obj_geometries)))
    depth_mat = np.zeros((len(obj_geometries), len(obj_geometries)))
    # overlap matrix for depth order
    is_overlap_matrix = np.zeros((len(obj_geometries), len(obj_geometries))) 
    obj_ids = list(obj_geometries.keys()) 


    cam_K = anno_cam["cam_K"]
    intrinsic = np.array(cam_K).reshape((3, 3))
    extrinsic = np.array([[1, 0, 0, 0],
                        [0, 1, 0 ,0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    render_width = 640
    ratio = render_width / img_w
    render_height = int(img_h * ratio)
    render.setup_camera(intrinsic, extrinsic, render_width, render_height)

    # set camera pose
    center = [0, 0, 1]  # look_at target
    eye = [0, 0, 0]  # camera position
    up = [0, -1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)
    # adjust intrinsic corresponding to resized image
    intrinsic[0, 0] = intrinsic[0, 0] * ratio
    intrinsic[0, 2] = intrinsic[0, 2] * ratio
    intrinsic[1, 1] = intrinsic[1, 1] * ratio
    intrinsic[1, 2] = intrinsic[1, 2] * ratio
    render.scene.camera.set_projection(intrinsic, 0.01, 4.0, render_width, render_height)

    print("Calculate occlusion & depth orders...")
    for idx_A in tqdm(obj_ids):
        
        amodal_mask = cv2.imread("{}/{:06d}_{:06d}.png".format(path_amodal_mask, im_id, idx_A))
        visib_mask = cv2.imread("{}/{:06d}_{:06d}.png".format(path_visible_mask, im_id, idx_A))
        visible_ratio = np.sum(visib_mask) / np.sum(amodal_mask)
        occ_cal = False if visible_ratio > 0.95 else True
            

        # set target i object as [1,0,0]
        obj_geometry = obj_geometries[idx_A]
        color = [1, 0, 0]
        obj_geometry.paint_uniform_color(color)
        render.scene.add_geometry(
                        "obj_{}".format(idx_A), obj_geometry, 
                        obj_mtl,  add_downsampled_copy_for_fast_rendering=True)
        mask_A = np.array(render.render_to_image())

        for idx_B in obj_ids:
            if idx_A == idx_B: continue
            # depth order
            A_s, A_e, B_s, B_e = obj_depths[idx_A][0], obj_depths[idx_A][1], obj_depths[idx_B][0], obj_depths[idx_B][1]
            if abs(A_s - B_s) < 0.01 and abs(A_e - B_e) < 0.01:
                depth_mat[idx_A, idx_B] = 1
                depth_mat[idx_B, idx_A] = 1
                is_overlap_matrix[idx_A, idx_B] = 1
                is_overlap_matrix[idx_B, idx_A] = 1
            elif A_e < B_s:
                depth_mat[idx_A, idx_B] = 1 # near, far
            elif B_e < A_s:
                depth_mat[idx_B, idx_A] = 1
            elif A_s < B_s and B_s < A_e:
                depth_mat[idx_A, idx_B] = 1
                is_overlap_matrix[idx_A, idx_B] = 1
            elif B_s < A_s and A_s < B_e:
                depth_mat[idx_B, idx_A] = 1
                is_overlap_matrix[idx_B, idx_A] = 1
            
            if not occ_cal: continue

            # set target j object as [0,0,1]
            obj_geometry = obj_geometries[idx_B]
            color = [0, 0, 0]
            obj_geometry.paint_uniform_color(color)
            render.scene.add_geometry(
                            "obj_{}".format(idx_B), obj_geometry, 
                            obj_mtl,  add_downsampled_copy_for_fast_rendering=True)
            mask_A_B = np.array(render.render_to_image())

            # count area
            
            cnd_r = mask_A[:, :, 0] != 0
            cnd_g = mask_A[:, :, 1] == 0
            cnd_b = mask_A[:, :, 2] == 0
            cnd_init = np.bitwise_and(np.bitwise_and(cnd_r, cnd_g), cnd_b)
            cnd_init = fill_hole(cnd_init)

            cnd_r = mask_A_B[:, :, 0] != 0
            cnd_g = mask_A_B[:, :, 1] == 0
            cnd_b = mask_A_B[:, :, 2] == 0
            cnd_sum = np.bitwise_and(np.bitwise_and(cnd_r, cnd_g), cnd_b)
            cnd_sum = fill_hole(cnd_sum)

            num_init = np.count_nonzero(cnd_init)
            num_sum = np.count_nonzero(cnd_sum)
            if num_init == 0 or num_sum == 0: 
                render.scene.clear_geometry()
                continue
            diff_rate = (num_init-num_sum) / num_init

            if diff_rate > 0.05:
                # print("OBJ {} - {} : {:.3f}%".format(idx_A, idx_B, diff_rate*100))
                occ_mat[idx_B, idx_A] = 1
            
            # revert the scene
            render.scene.remove_geometry("obj_{}".format(idx_B))
        render.scene.remove_geometry("obj_{}".format(idx_A))

    render.scene.clear_geometry()

    if os.path.exists(root_data + "/occ_mat.json"):
        with open(root_data + "/occ_mat.json", "r") as f:
            try:
                occ_mats = json.load(f)
            except:
                occ_mats = {}
        if isinstance(occ_mats, list):
            occ_mats = {}
    else:
        occ_mats = {}
    with open(root_data + "/occ_mat.json", "w") as f:
        occ_mats[str(int(im_id))] = np.array(occ_mat.reshape(-1), dtype=np.int).tolist()
        json.dump(occ_mats, f, indent=2)
        

    if os.path.exists(root_data + "/depth_mat.json"):
        with open(root_data + "/depth_mat.json", "r") as f:
            try:
                depth_mats = json.load(f)
            except:
                depth_mats = {}
        if isinstance(depth_mats, list):
            depth_mats = {}
    else:
        depth_mats = {}
    with open(root_data + "/depth_mat.json", "w+") as f:
        depth_mats[str(int(im_id))] = np.array(depth_mat.reshape(-1), dtype=np.float).tolist()
        json.dump(depth_mats, f, indent=2)
    
    if os.path.exists(root_data + "/is_overlap_matrix.json"):
        with open(root_data + "/is_overlap_matrix.json", "r") as f:
            try:
                is_overlap_matrixs = json.load(f)
            except:
                is_overlap_matrixs = {}
        if isinstance(is_overlap_matrixs, list):
            is_overlap_matrixs = {}
    else:
        is_overlap_matrixs = {}
    with open(root_data + "/is_overlap_matrix.json", "w+") as f:
        is_overlap_matrixs[str(int(im_id))] = np.array(is_overlap_matrix.reshape(-1), dtype=np.int).tolist()
        json.dump(is_overlap_matrixs, f, indent=2)
    # print("Occlusion order: {}".format(occ_mat))
    # print("Depth order: {}".format(depth_mat))