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
import json
import shutil

ood_root = os.environ['OOD_ROOT']
input_project_path = os.path.join(ood_root, "ours/data1/projects")
input_model_path = os.path.join(ood_root, "ours/data1/models_notaligned")
output_model_path = os.path.join(ood_root, "ours/data1/models")

transforms_not_aligned_to_aligned = {}
# Process models of all objects in the selected dataset.
for obj_id in range(1, 201):
  if obj_id < 100:
    continue
  project_in_path = os.path.join(input_project_path, "obj_{:06d}.mlp".format(obj_id))
  texture_in_path = os.path.join(input_model_path, "obj_{:06d}.png".format(obj_id))
  model_out_path = os.path.join(output_model_path, "obj_{:06d}.ply".format(obj_id))
  print("processing model: obj id: {}".format(obj_id))
  if not os.path.exists(project_in_path):
    transforms_not_aligned_to_aligned[obj_id] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    shutil.copy(os.path.join(input_model_path, "obj_{:06d}.ply".format(obj_id)), model_out_path)
    if os.path.exists(texture_in_path):
      shutil.copy(texture_in_path, os.path.join(output_model_path, "obj_{:06d}.png".format(obj_id)))
    continue

  with open(project_in_path, "r") as f:
    lines = f.readlines()
    read_lines = False
    transform = ""
    for line in lines:
      if "<MLMatrix44>" in line:
        read_lines = True
        continue
      if "</MLMatrix44>" in line:
        read_lines = False
        break
      if read_lines:
        transform += line
  transform = transform.replace("\n", ",").replace(" ", ",").replace("\t", ",")
  transform = transform.split(",")
  _transform = []
  for t in transform:
    if "e" in t:
      n1 = t.split("e")[0]
      n2 = t.split("e")[1]
      _transform.append(float(n1) * 10 ** float(n2))
      continue
    if t == '' or t == " " or t == "\n":
      continue
    try: 
      _transform.append(float(t))
    except:
      pass
  transform = _transform
  if len(transform) != 16:
    print("Error: invalid transform (len {}): {}".format(len(transform), transform))
    exit()
  with open(project_in_path, "r") as f:
    lines = f.readlines()
    _lines = []
    for line in lines:
      if "models_notaligned" in line:
        if "../models_notaligned" not in line:
          line = line.replace("models_notaligned", "../models_notaligned")
      _lines.append(line)
  with open(project_in_path, "w") as f:
    f.write("".join(_lines))
  transforms_not_aligned_to_aligned[obj_id] = transform
  texture_path = os.path.join(ood_root, "ours/data1/models_notaligned", "obj_{:06d}.png".format(int(obj_id)))
  ms = pymeshlab.MeshSet()
  ms.load_project(project_in_path)
  ms.apply_filter('apply_matrix_freeze', alllayers=True)
  if os.path.exists(texture_path):
    ms.save_current_mesh(model_out_path, save_vertex_color=False, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)
  else:
    ms.save_current_mesh(model_out_path, save_vertex_color=True, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)

with open(os.path.join(ood_root, "ours/data1/models_notaligned_to_models_aligned.json"), "w") as f:
  f.write(json.dumps(transforms_not_aligned_to_aligned, indent=4))
