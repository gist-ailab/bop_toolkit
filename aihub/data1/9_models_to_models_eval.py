# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""'Uniformly' resamples and decimates 3D object models for evaluation.
Note: Models of some T-LESS objects were processed by Blender (using the Remesh
modifier).
"""

import os
import glob
import pymeshlab
from pymeshlab import Percentage


ood_root = os.environ['OOD_ROOT']
input_model_path = os.path.join(ood_root, "ours/data1/models")
output_model_path = os.path.join(ood_root, "ours/data1/models_eval")

# Process models of all objects in the selected dataset.
for model_in_path in glob.glob(input_model_path + "/*.ply"):
  obj_id = os.path.basename(model_in_path).split("_")[-1].split(".")[0]
  print("Processing", model_in_path)
  model_out_path = os.path.join(output_model_path, os.path.basename(model_in_path))

  ms = pymeshlab.MeshSet()
  ms.load_new_mesh(model_in_path)
  ms.apply_filter('meshing_remove_unreferenced_vertices')
  ms.apply_filter('meshing_remove_duplicate_vertices')
  ms.apply_filter('meshing_remove_duplicate_faces')
  ms.apply_filter('generate_resampled_uniform_mesh', cellsize=Percentage(0.25), offset=Percentage(0.0), mergeclosevert=True, discretize=False, multisample=True, absdist=False)
  # ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=0, targetperc=0.025, qualitythr=0.5, preserveboundary=True, boundaryweight=1.0, preservenormal=True, preservetopology=False, optimalplacement=True, planarquadric=True, qualityweight=False, autoclean=True, selected=False)
  ms.save_current_mesh(model_out_path, save_vertex_color=True, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False)
  exit()
