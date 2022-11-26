# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the 3D bounding box and the diameter of 3D object models."""
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

import os
import glob
from tqdm import tqdm

ood_root = os.environ['OOD_ROOT']
input_model_path = os.path.join(ood_root, "ours/data1/models")

models_info = {}
for model_in_path in tqdm(glob.glob(input_model_path + "/*.ply")):
    
    obj_id = os.path.basename(model_in_path).split("_")[-1].split(".")[0]
    misc.log('Processing model of object {}...'.format(obj_id))
    model = inout.load_ply(model_in_path)

    # Calculate 3D bounding box.
    ref_pt = model['pts'].min(axis=0).flatten()
    size = (model['pts'].max(axis=0) - ref_pt)

    # map it to float
    ref_pt = list(map(float, ref_pt))
    size = list(map(float, size))



    # Calculated diameter.
    diameter = misc.calc_pts_diameter(model['pts'])

    models_info[obj_id] = {
        'min_x': ref_pt[0], 'min_y': ref_pt[1], 'min_z': ref_pt[2],
        'size_x': size[0], 'size_y': size[1], 'size_z': size[2],
        'diameter': diameter
    }

# Save the calculated info about the object models.
inout.save_json(os.path.join(ood_root, "ours/data1/models/models_info.json"), models_info)
