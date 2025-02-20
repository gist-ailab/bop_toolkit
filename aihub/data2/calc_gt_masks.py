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


    args = parser.parse_args()

    scene_id = args.scene_id
    im_id = args.im_id

    home_path = '/home/ailab'
    model_path = f"{home_path}/OccludedObjectDataset/ours/data1/models"

    dataset_path = f"{home_path}/OccludedObjectDataset/ours/data2/data2_real_source/all"
    img_id_range = range(1, 53)

    scene_gt_path = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_gt_{:06d}.json".format(scene_id))
    print("Process scene {} image {} \n".format(scene_id, im_id))
    root_data = os.path.join(dataset_path, "{:06d}".format(scene_id))
    path_rgb = os.path.join(root_data, 'rgb', '{:06d}.png')
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
    
    # set camera intrinsic
    cam_K = anno_cam["cam_K"]
    intrinsic = np.array(cam_K).reshape((3, 3))
    extrinsic = np.array([[1, 0, 0, 0],
                        [0, 1, 0 ,0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    render.setup_camera(intrinsic, extrinsic, img_w, img_h)

    # set camera pose
    center = [0, 0, 1]  # look_at target
    eye = [0, 0, 0]  # camera position
    up = [0, -1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)
    render.scene.camera.set_projection(intrinsic, 0.01, 4.0, img_w, img_h)

    # generate object material
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


    # generate amodal masks for all objects
    print("Calculate amodal masks ...")
    for obj_idx, obj_geometry in obj_geometries.items():
        # set color (two target and the others)
        color = [1, 0, 0]
        obj_geometry.paint_uniform_color(color)
        render.scene.add_geometry(
                        "obj_{}".format(obj_idx), obj_geometry, 
                        obj_mtl, add_downsampled_copy_for_fast_rendering=False)
        mask_init = np.array(render.render_to_image())
        
        cnd_r = mask_init[:, :, 0] != 0
        cnd_g = mask_init[:, :, 1] == 0
        cnd_b = mask_init[:, :, 2] == 0
        cnd_init = np.bitwise_and(np.bitwise_and(cnd_r, cnd_g), cnd_b)
        cnd_init = fill_hole(cnd_init)
        if np.sum(cnd_init) == 0:
            print('Empty mask is generating !!!!')
        # detect countour and delete small contours
        contours, _ = cv2.findContours(cnd_init.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                cv2.drawContours(cnd_init, [cnt], 0, 0, -1)
        cv2.imwrite("{}/{:06d}_{:06d}.png".format(path_amodal_mask, im_id, obj_idx), cnd_init.astype(np.uint8) * 255)
        render.scene.remove_geometry("obj_{}".format(obj_idx))

    # generate visible masks for all objects
    # firstly, load all objects 
    for obj_idx, obj_geometry in obj_geometries.items():
        # generate object material
        color = [0, 0, 0]
        obj_geometry.paint_uniform_color(color)
        render.scene.add_geometry(
                        "obj_{}".format(obj_idx), obj_geometry, 
                        obj_mtl, add_downsampled_copy_for_fast_rendering=False)
    
    # secondly, generate visible masks
    print("Calculate visible masks...")
    for obj_idx, obj_geometry in obj_geometries.items():
        render.scene.remove_geometry("obj_{}".format(obj_idx))
        # set red for the target object
        color = [1, 0, 0]
        obj_geometry.paint_uniform_color(color)
        render.scene.add_geometry(
                        "obj_{}".format(obj_idx), obj_geometry, 
                        obj_mtl, add_downsampled_copy_for_fast_rendering=False)
        mask_init = np.array(render.render_to_image())
        
        cnd_r = mask_init[:, :, 0] != 0
        cnd_g = mask_init[:, :, 1] == 0
        cnd_b = mask_init[:, :, 2] == 0
        cnd_init = np.bitwise_and(np.bitwise_and(cnd_r, cnd_g), cnd_b)
        cnd_init = fill_hole(cnd_init)
        contours, _ = cv2.findContours(cnd_init.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                cv2.drawContours(cnd_init, [cnt], 0, 0, -1)
        cv2.imwrite("{}/{:06d}_{:06d}.png".format(path_visible_mask, im_id, obj_idx), cnd_init.astype(np.uint8) * 255)
        render.scene.remove_geometry("obj_{}".format(obj_idx))
        color = [0, 0, 0]
        obj_geometry.paint_uniform_color(color)
        render.scene.add_geometry(
                        "obj_{}".format(obj_idx), obj_geometry, 
                        obj_mtl, add_downsampled_copy_for_fast_rendering=False)
    # remove all objects
    for obj_idx, obj_geometry in obj_geometries.items():
        render.scene.remove_geometry("obj_{}".format(obj_idx))

    