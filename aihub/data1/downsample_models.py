# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""'Uniformly' resamples and decimates 3D object models for evaluation.
Note: Models of some T-LESS objects were processed by Blender (using the Remesh
modifier).
"""

import os
import glob
import open3d as o3d
import numpy as np
from tqdm import tqdm

input_model_path = "/home/seung/OccludedObjectDataset/ours/data1/models_original"
output_model_path = "/home/seung/OccludedObjectDataset/ours/data1/models_anno"

# Attributes to save for the output models.
attrs_to_save = []

# Process models of all objects in the selected dataset.
for model_in_path in tqdm(glob.glob(input_model_path + "/*.ply")):

  id = int(os.path.basename(model_in_path).replace(".ply", "").split("_")[-1])
  if id!=115:
    continue
  print("processing model: obj id: {}".format(id))
  model_out_path = os.path.join(output_model_path, os.path.basename(model_in_path))
  pcd = o3d.io.read_point_cloud(model_in_path)
  pcd = pcd.voxel_down_sample(voxel_size=1)
  o3d.io.write_point_cloud(model_out_path, pcd)
