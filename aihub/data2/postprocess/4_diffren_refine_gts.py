import os, sys, glob
import json
import cv2
import numpy as np
from tqdm import tqdm
import copy
import open3d as o3d

treg = o3d.t.pipelines.registration 
import open3d as o3d
import numpy as np 
from tqdm import tqdm

import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# io utils
import pyredner
import matplotlib.pyplot as plt
import skimage
from aihub.data2.postprocess.utils import *

ood_root = os.environ['OOD_ROOT']
dataset_root = os.path.join(ood_root, 'ours/data2/data2_real_source/all')

scene_ids = list(range(1, 11))
im_ids = list(range(1, 53))


for scene_id in tqdm(scene_ids):
    scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_c1p1_{0:06d}.json".format(scene_id))
    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)
    scene_camera_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_camera.json")
    with open(scene_camera_path, "r") as f:
        scene_camera = json.load(f)
    
    new_scene_gt = {}
    for im_id in im_ids:
        im_gts = scene_gt[str(im_id)]
        new_im_gts = []

        # load rgb, depth 
        print("Loading rgb, depth images")
        rgb_path = os.path.join(dataset_root, f"{scene_id:06d}", "rgb", f"{im_id:06d}.png")
        rgb_im = cv2.imread(rgb_path)
        depth_path = os.path.join(dataset_root, f"{scene_id:06d}", "depth", f"{im_id:06d}.png")
        depth_im = cv2.imread(depth_path, -1)
        depth_scale = scene_camera[str(im_id)]['depth_scale']
        #!TODO: actually, depth scale should be multipled, not divided
        depth_im = depth_im.astype(np.float32)  / depth_scale
        
        # generate visible masks
        print("generating visible masks")
        height, width = rgb_im.shape[:2]
        cam_K = np.array(scene_camera[str(im_id)]['cam_K']).reshape(3, 3)
        visible_masks = gen_visible_masks(width, height, cam_K, im_gts)
        uoais_masks = run_uoais(rgb_im, depth_im)
        # reshape with (height, width)
        _uoais_masks = []
        for idx in range(uoais_masks.shape[0]):
            uoais_mask = np.uint8(uoais_masks[idx]*255)
            uoais_mask = cv2.resize(uoais_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            _uoais_masks.append(uoais_mask)
        uoais_masks = _uoais_masks

        for idx, obj_gt in enumerate(im_gts):

            if idx < 4:
                continue

            # load gts
            obj_id = obj_gt['obj_id']
            inst_id = obj_gt['inst_id']
            print("optimizing pose for obj_id {}, inst_id {}".format(obj_id, inst_id))

            H_c2m = np.eye(4)
            H_c2m[:3, :3] = np.array(obj_gt['cam_R_m2c']).copy().reshape(3, 3)
            H_c2m[:3, 3] = np.array(obj_gt['cam_t_m2c']).copy() 


            # load object meshes
            object_model_path = os.path.join(ood_root, f"ours/data1/models/obj_{obj_id:06d}.ply")
            object_mesh = o3d.io.read_triangle_mesh(object_model_path)
            object_mesh.transform(H_c2m)
            object_mesh.scale(0.001, [0, 0, 0])


            visible_mask = visible_masks["object_{}_{}".format(obj_id, inst_id)]

            # crop roi regions
            contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            w_offset = int(w * 0.1)
            h_offset = int(h * 0.1)
            roi_xyxy = [x-w_offset, y-h_offset, x+w+w_offset, y+h+h_offset]
            roi_xyxy = [max(0, roi_xyxy[0]), max(0, roi_xyxy[1]), min(width, roi_xyxy[2]), min(height, roi_xyxy[3])]

            # get roi's rgb, depth, mask
            rgb_roi = rgb_im[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2], :]
            depth_roi = depth_im[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2]]
            visible_mask_roi = visible_mask[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2]]


            # get the uoais mask with the best iou
            best_iou = 0
            for uoais_mask in uoais_masks:
                uoais_mask_roi = uoais_mask.copy()[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2]]
                iou = np.bitwise_and(uoais_mask_roi, visible_mask_roi).sum() / np.bitwise_or(uoais_mask_roi, visible_mask_roi).sum()
                if iou > best_iou:
                    best_iou = iou
                    mask_roi = uoais_mask_roi
            # invert mask_roi
            # mask_roi = np.uint8(np.where(mask_roi==0, 255, 0))

            # print("best iou: {}".format(best_iou))
            # print(np.unique(best_uoais_mask_roi))
            # cv2.imwrite("best_uoais_mask.png", best_uoais_mask_roi)

            pyredner.set_use_gpu(torch.cuda.is_available())
            diff_renderer = DiffRenderer(width, height, cam_K, roi_xyxy, obj_id, H_c2m)
        
            t_optimizer = torch.optim.SGD([diff_renderer.translation], lr=0.005, momentum=0.9)
            r_optimizer = torch.optim.SGD([diff_renderer.euler_angles], lr=0.005, momentum=0.9)

            rgb_roi_lcs = torch.tensor(np.power(skimage.img_as_float(rgb_roi[:, :, ::-1]).astype(np.float32), 2.2), device = pyredner.get_device())
            depth_roi_lcs =torch.tensor(np.power(skimage.img_as_float(depth_roi).astype(np.float32), 2.2), device = pyredner.get_device()).unsqueeze(-1)
            mask_roi_lcs = torch.tensor(np.power(skimage.img_as_float(mask_roi).astype(np.float32), 2.2), device = pyredner.get_device()).unsqueeze(-1)
            mask_roi_tensor = torch.tensor(mask_roi, device = pyredner.get_device()).unsqueeze(-1)
            imgs, losses = [], []
            # Run 80 Adam iterations
            num_iters = 1000
            best_loss = 1e10
            best_iter = 0
            for t in range(num_iters):
                t_optimizer.zero_grad()
                r_optimizer.zero_grad()

                render_imgs = diff_renderer.forward(['albedo', 'depth', 'mask'])
                albedo_loss = torch.abs((render_imgs['albedo'] - rgb_roi_lcs)*mask_roi_tensor).mean()
                depth_loss = torch.abs((render_imgs['depth'] - depth_roi_lcs)*mask_roi_tensor).mean()
                mask_loss = torch.abs(render_imgs['mask'] - mask_roi_lcs).mean() 
                loss = albedo_loss + depth_loss + mask_loss


                loss.backward()
                t_optimizer.step()
                r_optimizer.step()
                print(diff_renderer.translation, diff_renderer.euler_angles, loss.item())
                # Plot the loss
                print('iteration:', t, 'loss:', loss.item())
                losses.append(loss.data.item())
                if loss.data.item() < best_loss:
                    best_loss = loss.data.item()
                    best_render_imgs = render_imgs
                    best_iter = t
                    print("best iter: {}".format(best_iter))
                    best_translation = diff_renderer.translation.clone()
                    best_angles = diff_renderer.euler_angles.clone()

                if t % 100 == 0 and t!=0:
                    print("best iter: {}".format(best_iter))
                    print("best loss: {}".format(best_loss))
                    print("best translation: {}".format(best_translation))
                    print("best angles: {}".format(best_angles))
                    rgb_roi_render_vis = torch.pow(best_render_imgs['albedo'].clone(), 1.0/2.2).cpu().detach().numpy()
                    rgb_roi_vis = rgb_roi[:, :, ::-1]
                    depth_roi_render_vis = torch.pow(best_render_imgs['depth'].clone(), 1.0/2.2).cpu().detach().numpy()
                    depth_roi_vis = np.expand_dims(depth_roi.copy(), -1)
                    mask_roi_render_vis = torch.pow(best_render_imgs['mask'].clone(), 1.0/2.2).cpu().detach().numpy()
                    mask_roi_vis = np.expand_dims(mask_roi.copy(), -1)

                    f, (ax_loss, ax_img_1, ax_img_2, ax_img_3, ax_img_4, ax_img_5, ax_img_6, ax_img_7, ax_img_8, ax_img_9, ax_img_10) = plt.subplots(1, 11)
                    ax_loss.plot(range(len(losses)), losses, label='loss')
                    ax_loss.legend()
                    ax_img_1.imshow((rgb_roi_render_vis - rgb_roi_vis).sum(-1))
                    ax_img_2.imshow(rgb_roi_render_vis)
                    ax_img_3.imshow(rgb_roi_vis)
                    ax_img_4.imshow((depth_roi_render_vis - depth_roi_vis).sum(-1))
                    ax_img_5.imshow(depth_roi_render_vis)
                    ax_img_6.imshow(depth_roi_vis)
                    ax_img_7.imshow((mask_roi_render_vis - mask_roi_vis).sum(-1))
                    ax_img_8.imshow(mask_roi_render_vis)
                    ax_img_9.imshow(mask_roi_vis)
                    ax_img_10.imshow(np.uint8(np.where(mask_roi_render_vis==0, 0.5*rgb_roi_vis, rgb_roi_vis)))


                    # render_mask = np.where(rgb.data.cpu().numpy() == 0, 0, 255)
                    # rgb_roi_vis = torch.pow(rgb_roi_lcs.clone(), 1.0/2.2).cpu().numpy()
                    # overlay = np.uint8(np.where(render_mask == 0, rgb_roi_vis*0.5, rgb_roi_vis))
                    # ax_img_4.imshow(overlay)
                    plt.show()

                    # f, (ax_loss, ax_img_1, ax_img_2, ax_img_3, ax_img_4) = plt.subplots(1, 5)
                    # ax_loss.plot(range(len(losses)), losses, label='loss')
                    # ax_loss.legend()
                    # ax_img_1.imshow((depth.clone() - depth_roi).pow(2).sum(axis=2).data.cpu())
                    # ax_img_2.imshow(torch.pow(depth.data.clone(), 1.0/2.2).cpu())
                    # ax_img_3.imshow(torch.pow(depth_roi.clone(), 1.0/2.2).cpu())
                    # render_mask = np.where(depth.data.cpu().numpy() == 0, 0, 255)
                    # rgb_roi_vis = torch.pow(depth_roi.clone(), 1.0/2.2).cpu().numpy()
                    # overlay = np.uint8(np.where(render_mask == 0, rgb_roi_vis*0.5, rgb_roi_vis))
                    # ax_img_4.imshow(overlay)
                    # plt.show()

            exit()


            object_obj_path = "/home/seung/data/teapot.obj"
            verts, faces_idx, _ = load_obj(object_obj_path)
            faces = faces_idx.verts_idx
            # Initialize each vertex to be white in color.
            verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(device))
            # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
            object_mesh = Meshes(
                verts=[verts.to(device)],   
                faces=[faces.to(device)], 
                textures=textures
            )
            # initialize camera

            cam_R_m2c = np.array(obj_gt['cam_R_m2c']).reshape(3, 3)
            cam_t_m2c = np.array(obj_gt['cam_t_m2c'])
            H_cam2obj = np.eye(4)
            H_cam2obj[:3, :3] = cam_R_m2c
            H_cam2obj[:3, 3] = cam_t_m2c
            H_obj2cam = np.linalg.inv(H_cam2obj)
            R = torch.Tensor(H_obj2cam[:3, :3]).to(device).reshape(1, 3, 3)
            T = torch.Tensor(H_obj2cam[:3, 3]).to(device).reshape(1, 3, 1)
            # change it to 4x4 camera matrix
            # focal_length = torch.Tensor([cam_K_roi[0, 0], cam_K_roi[1, 1]]).to(device).reshape(1, 2)
            # principal_point = torch.Tensor([cam_K_roi[0, 2], cam_K_roi[1, 2]]).to(device).reshape(1, 2)
           
            camera = pyredner.Camera(
                intrinsic_mat=torch.Tensor(cam_K).to(device).reshape(3, 3),
                # cam_to_world=
            )

            exit()





            trans_init = np.identity(4)
            threshold = 0.004
            reg = o3d.pipelines.registration.registration_icp(object_pcd, scene_pcd, threshold, trans_init,
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                max_iteration=50))
            delta_H = reg.transformation.copy()              
            new_obj_gt = copy.deepcopy(obj_gt)   
            print(np.sum(np.abs(reg.transformation[:3, 3])))                              
            if np.sum(np.abs(reg.transformation[:3, 3])) < 0.25:
                delta_H[:3, 3] = delta_H[:3, 3] * 1000
                H_c2m = np.matmul(delta_H, H_c2m)
                new_obj_gt['cam_R_m2c'] = H_c2m[:3, :3].reshape(-1).tolist()
                new_obj_gt['cam_t_m2c'] = H_c2m[:3, 3].tolist()
                print("refine scene_id: {}, im_id: {}, obj_id: {}, inst_id: {}".format(scene_id, im_id, obj_id, inst_id))
            new_im_gts.append(new_obj_gt)
        new_scene_gt[str(im_id)] = new_im_gts
    new_scene_gt_path = os.path.join(dataset_root, f"{scene_id:06d}", "scene_gt_aligned_icp_{0:06d}.json".format(scene_id))
    with open(new_scene_gt_path, "w") as f:
        json.dump(new_scene_gt, f)



