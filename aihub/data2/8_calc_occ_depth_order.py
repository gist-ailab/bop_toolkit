import cv2
import json
import numpy as np
import open3d as o3d
import os
import pickle
import time
from tqdm import tqdm
import sys
sys.path.append('/home/seung/Workspace/papers/2022/clora/bop_renderer/build')

def floodfill_cnd(cnd_target, newVal, loDiff, upDiff):
    img_h, img_w = cnd_target.shape
    cnd_bg = np.zeros((img_h+2, img_w+2), dtype=np.uint8)
    cv2.floodFill(cnd_target.copy().astype(np.uint8), cnd_bg, (0,0), newVal, loDiff, upDiff)
    cnd_bg = cnd_bg[1:img_h+1, 1:img_w+1].astype(bool)
    cnd_target = 1 - cnd_bg.copy()
    return cnd_target


if __name__ == "__main__":

    ##############################
    # load model and annotations #
    ##############################
    dataset_path = "/home/seung/OccludedObjectDataset/ours/data2/data2_real_source/all"
    model_path = "/home/seung/OccludedObjectDataset/ours/data2/data2_real_source/models"
    # path
    scene_ids = sorted([int(x) for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))])
    # scene_ids = scene_ids[1:]
    for scene_id in tqdm(scene_ids):
        print("Process scene {}".format(scene_id))

        scene_gt = os.path.join(dataset_path, "{:06d}".format(scene_id), "scene_gt_{:06d}.json".format(scene_id))
        if not os.path.exists(scene_gt):
            print("Skip scene {} (GT file not found).".format(scene_id))
            continue

        for im_id in tqdm(range(1, 53)):
            root_data = os.path.join(dataset_path, "{:06d}".format(scene_id))
            path_rgb = os.path.join(root_data, 'rgb', '{:06d}.png')
            path_depth = os.path.join(root_data, 'depth', '{:06d}.png')
            path_anno_cam = os.path.join(root_data, 'scene_camera.json')
            path_anno_obj = os.path.join(root_data, 'scene_gt_{:06d}.json')
            

            # load camera pose annotation
            with open(path_anno_cam.format()) as gt_file:
                anno_cam = json.load(gt_file)
            anno_cam = anno_cam[str(im_id)] # type: dict

            # load object pose annotation
            with open(path_anno_obj.format(scene_id, scene_id)) as gt_file:
                anno_obj = json.load(gt_file)
            anno_obj = anno_obj[str(im_id)] # type: list

            # load object pointcloud and transform as annotation
            obj_geometries = {}
            obj_depths = {}
            for i, obj in enumerate(anno_obj):
                # print("... {} - {}_{}".format(i, obj['obj_id'], obj['inst_id']))
                # get transform
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
            rgb_img = cv2.imread(path_rgb.format(scene_id, im_id))
            img_h, img_w, img_c = rgb_img.shape
            render_width = img_w
            ratio = render_width / img_w
            render_height = int(img_h * ratio)
            print("Render size: {}x{}".format(render_width, render_height))
            render = o3d.visualization.rendering.OffscreenRenderer(
                                                width=render_width, height=render_height)
            # black background color
            render.scene.set_background([0, 0, 0, 1])
            render.scene.set_lighting(render.scene.LightingProfile.SOFT_SHADOWS, [0,0,0])
           
            # set camera intrinsic
            cam_K = anno_cam["cam_K"]
            cam_K[0] = cam_K[0] * ratio
            cam_K[2] = cam_K[2] * ratio
            cam_K[4] = cam_K[4] * ratio
            cam_K[5] = cam_K[5] * ratio
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
            render.scene.camera.set_projection(intrinsic, 0.01, 10.0, img_w, img_h)

            # generate occlusion order (n x n matrix)
            occ_mat = np.zeros((len(obj_geometries), len(obj_geometries)))
            depth_mat = np.zeros((len(obj_geometries), len(obj_geometries)))
            # overlap matrix for depth order
            is_overlap_matrix = np.zeros((len(obj_geometries), len(obj_geometries))) 
            obj_ids = list(obj_geometries.keys()) 
            obj_combs = ["{}_{}".format(i, j) for i in obj_ids for j in obj_ids]

            # set background objects as black
            for obj_idx, obj_geometry in obj_geometries.items():
                # generate object material
                obj_mtl = o3d.visualization.rendering.MaterialRecord()
                obj_mtl.base_color = [1.0, 1.0, 1.0, 1.0]
                obj_mtl.shader = "defaultUnlit"
                obj_mtl.point_size = 10.0
                # set color (two target and the others)
                color = [0, 0, 0]
                obj_geometry.paint_uniform_color(color)
                render.scene.add_geometry(
                                "background_obj_{}".format(obj_idx), obj_geometry, 
                                obj_mtl, add_downsampled_copy_for_fast_rendering=True)
            
            for obj_comb in tqdm(obj_combs):
                idx_A, idx_B = list(map(int, obj_comb.split("_")))

                if idx_A == idx_B: continue

                # depth order
                A_s, A_e, B_s, B_e = obj_depths[idx_A][0], obj_depths[idx_A][1], obj_depths[idx_B][0], obj_depths[idx_B][1]
                if A_e < B_s:
                    depth_mat[idx_A, idx_B] = 1 # near, far
                elif B_e < A_s:
                    depth_mat[idx_B, idx_A] = 1
                elif A_s < B_s and B_s < A_e:
                    depth_mat[idx_A, idx_B] = 1
                    is_overlap_matrix[idx_A, idx_B] = 1
                elif B_s < A_s and A_s < B_e:
                    depth_mat[idx_B, idx_A] = 1
                    is_overlap_matrix[idx_B, idx_A] = 1
                else:
                    print("ERROR: {} - {}".format(idx_A, idx_B))
                    exit()

                # occlusion order
                # check wheter both objects are not occluded by others, 
                # obj_id_A = int(anno_obj[idx_A]['obj_id'])
                # obj_id_B = int(anno_obj[idx_B]['obj_id'])
                # amodal_mask_A = cv2.imread('{}/mask/{:06d}_{:06d}.png'.format(root_data, im_id, idx_A), cv2.IMREAD_GRAYSCALE)
                # visible_mask_A = cv2.imread('{}/mask_visib/{:06d}_{:06d}.png'.format(root_data, im_id, idx_A), cv2.IMREAD_GRAYSCALE)
                # if np.count_nonzero(visible_mask_A) / np.count_nonzero(amodal_mask_A) > 0.99:
                #     continue
                # amodal_mask_B = cv2.imread('{}/mask/{:06d}_{:06d}.png'.format(root_data, im_id, idx_B), cv2.IMREAD_GRAYSCALE)
                # visible_mask_B = cv2.imread('{}/mask_visib/{:06d}_{:06d}.png'.format(root_data, im_id, idx_B), cv2.IMREAD_GRAYSCALE)
                # if np.count_nonzero(visible_mask_B) / np.count_nonzero(amodal_mask_B) > 0.99:
                #     continue

                # set target i object as [1,0,0]
                obj_geometry = obj_geometries[idx_A]
                color = [1, 0, 0]
                obj_geometry.paint_uniform_color(color)
                render.scene.remove_geometry("background_obj_{}".format(idx_A))
                render.scene.add_geometry(
                                "target_obj_{}".format(idx_A), obj_geometry, 
                                obj_mtl, add_downsampled_copy_for_fast_rendering=True)
                mask_init = np.array(render.render_to_image())
                            # get depth images

                # set target j object as [0,0,1]
                obj_geometry = obj_geometries[idx_B]
                color = [0, 0, 1]
                obj_geometry.paint_uniform_color(color)
                render.scene.remove_geometry("background_obj_{}".format(idx_B))
                render.scene.add_geometry(
                                "target_obj_{}".format(idx_B), obj_geometry, 
                                obj_mtl, add_downsampled_copy_for_fast_rendering=True)
                mask_sum = np.array(render.render_to_image())

                # count area
                newVal, loDiff, upDiff = 1, 1, 0
                cnd_r = mask_init[:, :, 0] != 0
                cnd_g = mask_init[:, :, 1] == 0
                cnd_b = mask_init[:, :, 2] == 0
                cnd_init = np.bitwise_and(np.bitwise_and(cnd_r, cnd_g), cnd_b)
                cnd_init = floodfill_cnd(cnd_init, newVal, loDiff, upDiff)

                cnd_r = mask_sum[:, :, 0] != 0
                cnd_g = mask_sum[:, :, 1] == 0
                cnd_b = mask_sum[:, :, 2] == 0
                cnd_sum = np.bitwise_and(np.bitwise_and(cnd_r, cnd_g), cnd_b)
                cnd_sum = floodfill_cnd(cnd_sum, newVal, loDiff, upDiff)
                cv2.imwrite("/home/seung/tmp/mask_init_{}_{}.png".format(idx_A, idx_B), cnd_init.astype(np.uint8)*255)
                cv2.imwrite("/home/seung/tmp/mask_sum_{}_{}.png".format(idx_A, idx_B), cnd_sum.astype(np.uint8)*255)
                # cv2.imwrite("mask_init.png", cnd_init.astype(np.uint8)*255)
                # cv2.imwrite("mask_sum.png", cnd_sum.astype(np.uint8)*255)

                num_init = np.count_nonzero(cnd_init)
                num_sum = np.count_nonzero(cnd_sum)
                if num_init == 0 or num_sum == 0: 
                    render.scene.clear_geometry()
                    continue
                diff_rate = (num_init-num_sum) / num_init

                if diff_rate > 0.05:
                    # print("OBJ {} - {} : {:.3f}%".format(idx_A, idx_B, diff_rate*100))
                    occ_mat[idx_A, idx_B] = 1
                
                
                # revert the scene
                render.scene.remove_geometry("target_obj_{}".format(idx_A))
                render.scene.remove_geometry("target_obj_{}".format(idx_B))
                obj_geometry = obj_geometries[idx_A]
                color = [0, 0, 0]
                obj_geometry.paint_uniform_color(color)
                render.scene.add_geometry(
                                "background_obj_{}".format(idx_A), obj_geometry, 
                                obj_mtl, add_downsampled_copy_for_fast_rendering=True)
                obj_geometry = obj_geometries[idx_B]
                obj_geometry.paint_uniform_color(color)
                render.scene.add_geometry(
                                "background_obj_{}".format(idx_B), obj_geometry, 
                                obj_mtl, add_downsampled_copy_for_fast_rendering=True)
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