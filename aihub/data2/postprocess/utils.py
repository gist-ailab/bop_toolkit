import open3d as o3d
import numpy as np
import pyredner
import torch
import os
import glob
import multiprocessing as mp
import os
import cv2
import imageio
import random
import numpy as np
# constants
from detectron2.engine import DefaultPredictor

from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

def normalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm) 
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth

def unnormalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ unnormalize the input depth (0 ~ 255) and return depth image (mm)
    Args:
        depth([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.
    Returns:
        [np.float]: depth array [H, W] (mm) 
    """
    depth = np.float32(depth) / 255
    depth = depth * (max_val - min_val) + min_val
    return depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """ inpaint the input depth where the value is equal to zero

    Args:
        depth ([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        factor (int, optional): resize factor in depth inpainting. Defaults to 4.
        kernel_size (int, optional): kernel size in depth inpainting. Defaults to 5.

    Returns:
        [np.uint8]: inpainted depth array [H, W, 3] (0 ~ 255)
    """
    
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W//factor, H//factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth

def load_segm_model():

    model = smp.create_model(
            arch = "DeepLabV3",
            encoder_name = "resnet50",
            in_channels = 3,
            classes = 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using {} gpu!".format(torch.cuda.device_count()))
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load("/home/seung/Workspace/papers/2022/clora/bop_toolkit/aihub/data2/foreground_segmentation/output/itr_53000.pkl")
    model.load_state_dict(checkpoint)
    model = model.eval()
    return model

from torchvision.transforms import Normalize, Compose
normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def infer_segm_mask(segm_model, rgb):

    device = pyredner.get_device()
    h, w = rgb.shape[:2]
    input = cv2.resize(rgb.copy(), (224, 224))
    input = torch.tensor(np.transpose(input, (2, 0, 1)), dtype=torch.float32)
    input = normalize(input).unsqueeze(0)
    input = input.to(device)
    pred = segm_model(input)
    pred = torch.sigmoid(pred)
    pred = torch.argmax(pred, dim=1).unsqueeze(0)
    # resize
    pred = pred[0].permute(1, 2, 0)
    pred = pred.detach().cpu().numpy()
    pred = np.uint8(np.repeat(pred, 3, axis=-1)*255)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
    pred = np.uint8(pred > 128)
    # print(rgb.shape, pred.shape)
    # pred_on_input = cv2.addWeighted(rgb, 0.5, pred, 0.5, 0)

    # vis_all = np.hstack([rgb, pred_on_input, pred])
    return pred



ood_root = os.environ['OOD_ROOT']

def gen_visible_masks(width, height, cam_K, im_gts):
    masks = {}
    render = o3d.visualization.rendering.OffscreenRenderer(width=width, height=height)
    render.scene.set_background([0, 0, 0, 1])
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, [0,0,0])
    intrinsic = np.array(cam_K).reshape((3, 3))
    extrinsic = np.eye(4)
    render.setup_camera(intrinsic, extrinsic, width, height)
    # set camera pose
    center = [0, 0, 1]  # look_at target
    eye = [0, 0, 0]  # camera position
    up = [0, -1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)
    render.scene.camera.set_projection(intrinsic, 0.01, 4.0, width, height)
    
    target_obj_mtl = o3d.visualization.rendering.MaterialRecord()
    target_obj_mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    target_obj_mtl.shader = "defaultUnlit"
    other_obj_mtl = o3d.visualization.rendering.MaterialRecord()
    other_obj_mtl.base_color = [0.0, 0.0, 0.0, 1.0]
    other_obj_mtl.shader = "defaultUnlit"

    # load all meshes
    object_meshes = {}
    for obj_gt in im_gts:
        obj_id = obj_gt['obj_id']
        inst_id = obj_gt['inst_id']
        H_c2m = np.eye(4)
        H_c2m[:3, :3] = np.array(obj_gt['cam_R_m2c']).copy().reshape(3, 3)
        H_c2m[:3, 3] = np.array(obj_gt['cam_t_m2c']).copy() 
        object_model_path = os.path.join(ood_root, f"ours/data1/models/obj_{obj_id:06d}.ply")
        object_mesh = o3d.io.read_triangle_mesh(object_model_path)
        object_mesh.transform(H_c2m)
        object_mesh.scale(0.001, [0, 0, 0])
        object_meshes["{}_{}".format(obj_id, inst_id)] = object_mesh
        render.scene.add_geometry("object_{}_{}".format(obj_id, inst_id), object_mesh, other_obj_mtl)
    
    masks = {}
    for obj_gt in im_gts:
        render.scene.remove_geometry("object_{}_{}".format(obj_gt['obj_id'], obj_gt['inst_id']))
        render.scene.add_geometry("object_{}_{}".format(obj_gt['obj_id'], obj_gt['inst_id']), object_meshes["{}_{}".format(obj_gt['obj_id'], obj_gt['inst_id'])], target_obj_mtl)
        mask_init = np.array(render.render_to_image())
        mask = np.where(mask_init[:, :, 0] > 125, 255, 0)
        mask = mask.astype(np.uint8)
        masks["object_{}_{}".format(obj_gt['obj_id'], obj_gt['inst_id'])] = mask
        render.scene.remove_geometry("object_{}_{}".format(obj_gt['obj_id'], obj_gt['inst_id']))
        render.scene.add_geometry("object_{}_{}".format(obj_gt['obj_id'], obj_gt['inst_id']), object_meshes["{}_{}".format(obj_gt['obj_id'], obj_gt['inst_id'])], other_obj_mtl)
    return masks

def gen_amodal_mask(width, height, cam_K, object_mesh):
    render = o3d.visualization.rendering.OffscreenRenderer(width=width, height=height)
    render.scene.set_background([0, 0, 0, 1])
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, [0,0,0])
    intrinsic = np.array(cam_K).reshape((3, 3))
    extrinsic = np.eye(4)
    render.setup_camera(intrinsic, extrinsic, width, height)
    # set camera pose
    center = [0, 0, 1]  # look_at target
    eye = [0, 0, 0]  # camera position
    up = [0, -1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)
    render.scene.camera.set_projection(intrinsic, 0.01, 4.0, width, height)
    obj_mtl = o3d.visualization.rendering.MaterialRecord()
    obj_mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    obj_mtl.shader = "defaultUnlit"
    # obj_mtl.point_size = 1.0
    render.scene.add_geometry("object", object_mesh, obj_mtl)
    mask_init = np.array(render.render_to_image())
    # get bounding box
    mask = np.where(mask_init[:, :, 0] > 125, 255, 0)
    mask = mask.astype(np.uint8)
    return mask

def run_uoais(rgb_img, depth_img):

    # UOAIS-Net
    cfg = get_cfg()
    cfg.merge_from_file('/home/seung/Workspace/papers/2021/UOAIS/uoais/configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml')
    cfg.defrost()
    cfg.MODEL.WEIGHTS = "/home/seung/Workspace/papers/2021/UOAIS/output/R50_rgbdconcat_mlc_occatmask_hom_concat/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.00001
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.9999
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    rgb_img = cv2.resize(rgb_img, (W, H))
    depth_img = depth_img.copy()

    rgb_img = cv2.resize(rgb_img, (W, H))
    depth_img = normalize_depth(depth_img)
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)
    uoais_input = np.concatenate([rgb_img, depth_img], -1)        
    # uoais_input = rgb_img
    outputs = predictor(uoais_input)
    instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

    preds = instances.pred_masks.detach().cpu().numpy() 
    pred_visibles = instances.pred_visible_masks.detach().cpu().numpy() 
    bboxes = instances.pred_boxes.tensor.detach().cpu().numpy() 
    pred_occs = instances.pred_occlusions.detach().cpu().numpy() 
   
    # reorder predictions for visualization
    idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
    preds, pred_occs, bboxes = preds[idx_shuf], pred_occs[idx_shuf], bboxes[idx_shuf]
    vis_img = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)
    vis_all_img = np.hstack([rgb_img, depth_img, vis_img])

    cv2.imwrite("test.png", vis_all_img)
    return pred_visibles
    exit()   


class DiffRenderer:

    def __init__(self, width, height, cam_K, viewport, obj_id, H_c2m) -> None:
        
         # adjust cam_K correspondingly
        normalized_cam_K = np.eye(3)
        normalized_cam_K[0, 0] = 2*cam_K[0, 0] / width
        normalized_cam_K[1, 1] = 2*cam_K[0, 0] / width
        normalized_cam_K[0, 2] = - (width - 2 * cam_K[0, 2]) / width
        
         # load camera for differentiable rendering
        self.camera = pyredner.Camera(
            intrinsic_mat=torch.Tensor(normalized_cam_K),
            position=torch.Tensor([0, 0, 0]),
            up = torch.Tensor([0, -1, 0]),
            look_at= torch.Tensor([0, 0, 1]),
            resolution=(height, width),
            viewport=(viewport[1], viewport[0], viewport[3], viewport[2]),
        )

        object_obj_path = os.path.join(ood_root, f"ours/data1/models_obj/obj_{obj_id:06d}.obj")
        objects_ori = pyredner.load_obj(object_obj_path, return_objects=True, device=pyredner.get_device())
        objects_tmp = pyredner.load_obj(object_obj_path, return_objects=True, device=pyredner.get_device())
        self.obj_ori = objects_ori[0]
        self.obj_tmp = objects_tmp[0]

        self.H_c2m = torch.tensor(H_c2m, device=pyredner.get_device(), dtype=torch.float32)
        self.T_c2m = self.H_c2m[:3, 3].clone()
        self.obj_ori.vertices = torch.matmul(torch.tensor(self.obj_ori.vertices, dtype=torch.float32, device=pyredner.get_device()), torch.t(self.H_c2m[:3, :3].clone()))  + self.H_c2m[:3, 3].clone()
        # self.obj_tmp.vertices = torch.matmul(torch.tensor(self.obj_tmp.vertices, dtype=torch.float32, device=pyredner.get_device()), torch.t(H_c2m[:3, :3].clone()))  + H_c2m[:3, 3].clone()

        self.translation = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device(), requires_grad=True)
        self.euler_angles = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device(), requires_grad=True)

    def forward(self, targets=['albedo']):
        
        rotation_matrix = pyredner.gen_rotate_matrix(self.euler_angles)
        self.obj_tmp.vertices = torch.matmul(self.obj_ori.vertices.clone()-self.T_c2m.clone(), rotation_matrix) + self.T_c2m.clone() + self.translation
        # self.obj.vertices = self.obj.vertices + self.translation

        scene = pyredner.Scene(camera = self.camera, objects = [self.obj_tmp])
        # Render the scene.
        imgs = {}
        for target in targets:
            if target == 'albedo':
                img = pyredner.render_albedo(scene)
            elif target == 'depth':
                img = pyredner.render_g_buffer(scene, channels= [pyredner.channels.depth])
            elif target == 'mask':
                img = pyredner.render_g_buffer(scene, channels= [pyredner.channels.alpha])
            imgs[target] = img
        return imgs


def dice_loss(img1, img2):
    # img1: torch tensor, img2: torch tensor
    # calculate dice loss
    smooth = 1.
    img1 = img1.view(-1)
    img2 = img2.view(-1)
    intersection = (img1 * img2).sum()
    return 1 - ((2. * intersection + smooth) / (img1.sum() + img2.sum() + smooth))

