# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""'Uniformly' resamples and decimates 3D object models for evaluation.
Note: Models of some T-LESS objects were processed by Blender (using the Remesh
modifier).
"""

import os
import glob
import pymeshlab
import pandas as pd
import shutil
from tqdm import tqdm

ood_root = os.environ['OOD_ROOT']
models_info = pd.read_excel("./assets/models_info.ods")


ood_root = os.environ['OOD_ROOT']
input_model_path = os.path.join(ood_root, "ours/data1/models_fine")
output_model_path = os.path.join(ood_root, "ours/data1/models")

existing_obj_ids = []
for path in glob.glob(os.path.join(output_model_path, "*.ply")):
    existing_obj_ids.append(int(os.path.basename(path).split("_")[1].split(".")[0]))


# Process models of all objects in the selected dataset.
for model_in_path in tqdm(sorted(glob.glob(input_model_path + "/*.ply"))):
  obj_id = os.path.basename(model_in_path).split("_")[-1].split(".")[0]
  # if int(obj_id) in existing_obj_ids:
    # continue
  model_out_path = os.path.join(output_model_path, os.path.basename(model_in_path))
  texture_path = model_in_path.replace('ply', 'png')
  texture_out_path = model_out_path.replace('ply', 'png')
  print("Processing", model_in_path)
  mb = os.path.getsize(model_in_path) / 1024 / 1024
  if mb < 5:
    shutil.copy(model_in_path, model_out_path)
    if os.path.exists(texture_path):
      shutil.copy(texture_path, texture_out_path)
  else:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_in_path)
    if os.path.exists(texture_path):
      ms.apply_filter('compute_texcoord_transfer_vertex_to_wedge')
      ms.apply_filter('compute_color_from_texture_per_vertex')
      ms.apply_filter('generate_simplified_point_cloud', samplenum = 50000, exactnumflag= True)
      ms.apply_filter('remove_duplicate_vertices')
      ms.apply_filter('generate_surface_reconstruction_screened_poisson', preclean=True)
      ms.save_current_mesh(model_out_path, save_vertex_color=True, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)
    else:
      ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=15728, qualitythr=0.5, preserveboundary=True, boundaryweight=1.0, preservenormal=True, optimalplacement=True, planarquadric=True, selected=False)
      ms.save_current_mesh(model_out_path, save_vertex_color=True, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)
