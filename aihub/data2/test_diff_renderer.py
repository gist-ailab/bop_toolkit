import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import json 

# io utils
from pytorch3d.io import load_obj, load_ply

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)


ply_path = "/OccludedObjectDataset/ours/data1/models/000001.ply"
data_root = '/OccludedObjectDataset/ours/data2/data2_real_source/all'
model_root = "/OccludedObjectDataset/ours/data1/models"

scene_id = 28
im_id = 1

scene_path = os.path.join(data_root, "{:06d}".format(scene_id))
scene_gt_path = os.path.join(scene_path, "scene_gt_{:06d}.json".format(scene_id))
scene_gt_info = json.load(open(scene_gt_path, 'r'))
scene_camera_path = os.path.join(scene_path, "scene_camera.json")
scene_camera_info = json.load(open(scene_camera_path, 'r'))

im_gt_info = scene_gt_info[str(im_id)]
im_path = os.path.join(scene_path, "rgb", "{:06d}.png".format(im_id))
im_camera_info = scene_camera_info[str(im_id)]

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

for im_gt in im_gt_info:
    obj_id = im_gt['obj_id']
    if obj_id != 1:
        continue
    
    # ply_path = os.path.join(model_root, "obj_{:06d}.ply".format(obj_id))
    obj_path = "/home/seung/OccludedObjectDataset/ours/data1/models_obj/obj_000001.obj"
    cam_R_m2c = np.array(im_gt['cam_R_m2c']).reshape(3, 3)
    cam_t_m2c = np.array(im_gt['cam_t_m2c']).reshape(3, 1)
    cam_K = np.array(im_camera_info['cam_K']).reshape(3, 3)

    verts, faces_idx, _ = load_obj(obj_path)
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    teapot_mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], 
        textures=textures
    )

    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    # We can add a point light in front of the object. 
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )


