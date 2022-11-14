# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""'Uniformly' resamples and decimates 3D object models for evaluation.

Note: Models of some T-LESS objects were processed by Blender (using the Remesh
modifier).
"""

import os

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import misc
import glob

# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'lm',

  # Type of input object models.
  # None = default model type.
  'model_in_type': None,

  # Type of output object models.
  'model_out_type': 'eval',

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # Path to meshlabserver.exe (tested version: 1.3.3).
  # On Windows: C:\Program Files\VCG\MeshLab133\meshlabserver.exe
  'meshlab_server_path': '/usr/bin/meshlabserver',

  # Path to scripts/meshlab_scripts/remesh_for_eval.mlx.
  'meshlab_script_path': '/home/seung/Workspace/papers/2022/clora/bop_toolkit/scripts/meshlab_scripts/remesh_for_eval_cell=0.25.mlx',
}
################################################################################

ood_root = os.environ['OOD_ROOT']
# Load dataset parameters.
dp_model_in = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], p['model_in_type'])

dp_model_out = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], p['model_out_type'])

input_model_path = os.path.join(ood_root, "ours/data1/models")
output_model_path = os.path.join(ood_root, "ours/data1/models_eval")

# Attributes to save for the output models.
attrs_to_save = []
# Process models of all objects in the selected dataset.
for model_in_path in sorted(glob.glob(input_model_path + "/*.ply"), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1])):
    obj_id = os.path.basename(model_in_path).split("_")[-1].split(".")[0]
    if int(obj_id) in [6, 9] or int(obj_id) < 10:
      continue
    misc.log('\n\n\nProcessing model of object {}...\n'.format(obj_id))
    model_out_path = os.path.join(output_model_path, os.path.basename(model_in_path))
    misc.ensure_dir(os.path.dirname(model_out_path))
    misc.run_meshlab_script(p['meshlab_server_path'], p['meshlab_script_path'], model_in_path, model_out_path, attrs_to_save)

misc.log('Done.')
