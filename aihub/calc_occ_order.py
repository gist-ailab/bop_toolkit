import cv2
import json
import numpy as np
import open3d as o3d
import os
import pickle

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
    dataset_path = "/home/seung/OccludedObjectDataset/data2_source"
    model_path = "/home/seung/OccludedObjectDataset/models"
    # path
    scene_ids = [15, 38, 39]
    for scene_id in scene_ids:
        for im_id in range(1, 53):
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
                obj_geometry.translate(transform_cam_to_obj[0:3, 3])
                center = obj_geometry.get_center()
                obj_geometry.rotate(transform_cam_to_obj[0:3, 0:3], center=center)
                # save in dictionary
                obj_geometries[i] = obj_geometry

            ###############################
            # generate offscreen renderer #
            ###############################
            # generate offscreen renderer
            rgb_img = cv2.imread(path_rgb.format(scene_id, im_id))
            img_h, img_w, img_c = rgb_img.shape
            render = o3d.visualization.rendering.OffscreenRenderer(
                                                width=img_w, height=img_h)
            # black background color
            render.scene.set_background([0, 0, 0, 1])
            render.scene.set_lighting(render.scene.LightingProfile.SOFT_SHADOWS, [0,0,0])
            # generate object material
            obj_mtl = o3d.visualization.rendering.MaterialRecord()
            obj_mtl.base_color = [1.0, 1.0, 1.0, 1.0]
            obj_mtl.shader = "defaultUnlit"
            # set camera intrinsic
            cam_K = anno_cam["cam_K"]
            intrinsic = np.array(cam_K).reshape((3, 3))
            extrinsic = np.array([[1, 0, 0, 0],
                                [0, 1, 0 ,0],
                                [0, 0, 1, -0.13128653],
                                [0, 0, 0, 1]])
            render.setup_camera(intrinsic, extrinsic, img_w, img_h)
            # set camera pose
            center = [0, 0, 1]  # look_at target
            eye = [0, 0, 0]  # camera position
            up = [0, -1, 0]  # camera orientation
            render.scene.camera.look_at(center, eye, up)
            render.scene.camera.set_projection(intrinsic, 0.01, 3.0, img_w, img_h)

            # generate occlusion order (n x n matrix)
            occ_mat = np.zeros((len(obj_geometries), len(obj_geometries)))
            obj_ids = list(obj_geometries.keys()) 
            obj_combs = ["{}_{}".format(i, j) for i in obj_ids for j in obj_ids]
            for obj_comb in obj_combs:
                idx_i, idx_j = list(map(int, obj_comb.split("_")))
                if idx_i == idx_j: continue

                # set background objects as black
                for obj_idx, obj_geometry in obj_geometries.items():
                    # set color (two target and the others)
                    if obj_idx in [idx_i, idx_j]: continue
                    color = [0, 0, 0]
                    obj_geometry.paint_uniform_color(color)
                    render.scene.add_geometry(
                                    "background_obj_{}".format(obj_idx), obj_geometry, 
                                    obj_mtl, add_downsampled_copy_for_fast_rendering=False)

                # set target i object as [1,0,0]
                obj_geometry = obj_geometries[idx_i]
                color = [1, 0, 0]
                obj_geometry.paint_uniform_color(color)
                render.scene.add_geometry(
                                "target_obj_{}".format(idx_i), obj_geometry, 
                                obj_mtl, add_downsampled_copy_for_fast_rendering=False)
                mask_init = np.array(render.render_to_image())

                # set target j object as [0,0,1]
                obj_geometry = obj_geometries[idx_j]
                color = [0, 0, 1]
                obj_geometry.paint_uniform_color(color)
                render.scene.add_geometry(
                                "target_obj_{}".format(idx_j), obj_geometry, 
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

                num_init = np.count_nonzero(cnd_init)
                num_sum = np.count_nonzero(cnd_sum)
                if num_init == 0 or num_sum == 0: 
                    print("num_init: {}, num_sum: {}".format(num_init, num_sum))
                    render.scene.clear_geometry()
                    continue
                diff_rate = (num_init-num_sum) / num_init

                if diff_rate > 0.05:
                    print("OBJ {} - {} : {:.3f}%".format(idx_i, idx_j, diff_rate*100))
                    occ_mat[idx_i, idx_j] = 1

                render.scene.clear_geometry()
            
            with open(root_data + "/occ_mat.json", "w") as f:
                occ_mat = np.array(occ_mat, dtype=np.int)
                json.dump(occ_mat.tolist(), f)
