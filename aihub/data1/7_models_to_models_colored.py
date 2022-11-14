#! TODO: problem in saving only colored vertex

import pymeshlab
import os
import glob
from tqdm import tqdm

ood_root = os.environ['OOD_ROOT']
input_model_path = os.path.join(ood_root, "ours/data1/models")
output_model_path = os.path.join(ood_root, "ours/data1/models_colored")

# Attributes to save for the output models.
attrs_to_save = []

# Process models of all objects in the selected dataset.
for model_in_path in tqdm(glob.glob(input_model_path + "/*.ply")):

  model_out_path = os.path.join(output_model_path, os.path.basename(model_in_path))
  id = int(os.path.basename(model_in_path).replace(".ply", "").split("_")[-1])
  ms = pymeshlab.MeshSet()
  ms.load_new_mesh(model_in_path)
  ms.apply_filter('compute_color_from_texture_per_vertex')
  ms.save_current_mesh(model_out_path, save_vertex_color=True, save_vertex_normal=False, save_vertax_flag=False, save_face_color=False, save_wedge_texcoord=False, save_wedge_color=False, binary=False)