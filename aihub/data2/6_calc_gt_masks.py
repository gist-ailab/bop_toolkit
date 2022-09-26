# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates masks of object models in the ground-truth poses."""

import os
import numpy as np
import sys

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visibility

from tqdm import tqdm
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'data2_real_source',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'all',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # Tolerance used in the visibility test [mm].
  'delta': 15,  # 5 for ITODD, 15 for the other datasets.

  # Type of the renderer.
  'renderer_type': 'cpp',  # Options: 'vispy', 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': '/home/seung/OccludedObjectDataset/ours/data2'
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params_clora(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = None
if p['dataset'] == 'tless':
  model_type = 'cad'
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
# scene_ids = [x for x in scene_ids if x > 158]
scene_ids = [1]

for scene_id in tqdm(scene_ids):

  # Load scene GT.
  scene_gt_path = dp_split['scene_gt_tpath'].format(
    scene_id=scene_id)
  if not os.path.exists(scene_gt_path):
    print("Skip scene {} (GT file not found).".format(scene_id))
    continue
  else:
    print("Process scene {}".format(scene_id))
  scene_gt = inout.load_scene_gt(scene_gt_path)

  # Load scene camera.
  scene_camera_path = dp_split['scene_camera_tpath'].format(
    scene_id=scene_id)
  scene_camera = inout.load_scene_camera(scene_camera_path)

  # Create folders for the output masks (if they do not exist yet).
  mask_dir_path = os.path.dirname(
    dp_split['mask_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_dir_path)

  mask_visib_dir_path = os.path.dirname(
    dp_split['mask_visib_tpath'].format(
      scene_id=scene_id, im_id=0, gt_id=0))
  misc.ensure_dir(mask_visib_dir_path)

  # Initialize a renderer.
  misc.log('Initializing renderer...')
  #!TODO: test on cpp renderer
  #!TODO: run for each camera
  for camera_idx in tqdm(range(4)):
    if camera_idx == 1 or camera_idx == 2:
      width, height = 1920, 1080
    elif camera_idx == 3:
      width, height = 3840, 2160
    elif camera_idx == 0:
      width, height = 1920, 1200

    # ren = renderer.create_renderer(
      # width, height, renderer_type=p['renderer_type'], mode='depth')
    


    # Add object models.
    # for obj_id in dp_model['obj_ids']:
    #   mesh = pyrender.Mesh.from_points(dp_model['model_tpath'].format(obj_id=obj_id))
    #   scene.add(mesh)
      # ren.add_object(obj_id, dp_model['model_tpath'].format(obj_id=obj_id))
    im_ids = sorted(scene_gt.keys())
    im_ids = [x for x in im_ids if x > 0 and x % 4 == camera_idx]

    for im_id in tqdm(im_ids):
      if im_id % 100 == 0:
        misc.log(
          'Calculating masks - dataset: {} ({}, {}), scene: {}, im: {}'.format(
            p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id,
            im_id))
      K = scene_camera[im_id]['cam_K']
      fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
      
      # fuze_trimesh = trimesh.load('/home/seung/OccludedObjectDataset/ours/data2/data2_real_source/models/obj_000022.ply')
      # mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
      # scene = pyrender.Scene()
      # scene.add(mesh)
      # pyrender.Viewer(scene, use_raymond_lighting=True)


      # render amodal masks for each objects
      for gt_id, gt in enumerate(scene_gt[im_id]):
        print(dp_model['model_tpath'].format(obj_id=gt['obj_id']))

        scene = pyrender.Scene()

        mesh_tri = trimesh.load(dp_model['model_tpath'].format(obj_id=gt['obj_id']))
        mesh_tri = mesh_tri.apply_scale(0.001)
        # mesh_tri.visual.vertex_colors = [0, 0, 0, 0]
        # mesh_tri = mesh_tri. (0.001)
       
        # pyrender.Viewer(scene, use_raymond_lighting=True)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        H = np.eye(4)
        H[:3, :3] = scene_camera[im_id]['cam_R_w2c']
        H[:3, 3] = np.array(scene_camera[im_id]['cam_t_w2c']).reshape(3) / 1000
        # convert to opengl coordinate
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R.from_euler('x', 180, degrees=True).as_matrix()
        camera_pose = np.matmul(camera_pose, H)
        print(camera_pose, H)
        scene.add(camera, pose=camera_pose)

        mesh = pyrender.Mesh.from_trimesh(mesh_tri)
        object_pose = np.eye(4)
        object_pose[:3, :3] = gt['cam_R_m2c']
        object_pose[:3, 3] = np.array(gt['cam_t_m2c']).reshape(3)
        H_w2m = np.matmul(camera_pose, np.linalg.inv(object_pose))
        scene.add(mesh, pose=object_pose)

        # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
        #                     innerConeAngle=np.pi/16.0,
        #                     outerConeAngle=np.pi/6.0)
        # scene.add(light, pose=H)


        # object_node = pyrender.Node(mesh=mesh, matrix=object_pose)
        # object_node = pyrender.Node(mesh=mesh)
        # scene.add_node(object_node)
        # scene.add(mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True)

        ren = pyrender.OffscreenRenderer(width, height)
        amodal_mask, depth = ren.render(scene, use_raymond_lighting=True)
        


        nm = {node: (i + 1) for i, node in enumerate(scene.mesh_nodes)}   # Node->Seg Id map
        print(nm)
        # amodal_mask = ren.render(scene, pyrender.RenderFlags.SEG, nm)[0]
        # amodal_mask = ren.render(scene)[0]

        # amodal_mask = np.uint8(np.where(amodal_mask != 0, 1, 0)) * 255
        print("==>", np.unique(amodal_mask))
        # Save the calculated masks.
        mask_path = dp_split['mask_tpath'].format(
          scene_id=scene_id, im_id=im_id, gt_id=gt_id)
        inout.save_im(mask_path, amodal_mask)
        scene.remove_node(object_node)
        exit()

        # mask_visib_path = dp_split['mask_visib_tpath'].format(
        #   scene_id=scene_id, im_id=im_id, gt_id=gt_id)
        # inout.save_im(mask_visib_path, 255 * mask_visib.astype(np.uint8))
