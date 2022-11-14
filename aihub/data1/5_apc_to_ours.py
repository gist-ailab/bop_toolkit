import os
import glob
import pymeshlab
from tqdm import tqdm
import cv2
import os
import pandas as pd
from tqdm import tqdm
ood_root = os.environ['OOD_ROOT']


apc_models_path = os.path.join(ood_root, "ours/data1/apc_main/object_models/tarball")
models_path = os.path.join(ood_root, "ours/data1/models")
models_info = pd.read_excel("./assets/models_info.ods")

object_ids = []
apc_names = []
for object_id, apc_name, paper in zip(models_info["object_id"], models_info["original name"], models_info['논문']):
    try: 
        int(object_id)
    except:
        continue
    else:
        if not isinstance(paper, str):
            continue
        if paper == 'APC':
            object_ids.append(int(object_id))
            apc_names.append(apc_name)

for object_id, apc_name in zip(tqdm(object_ids), apc_names):

    print('processing object_id: {}, apc_name: {}'.format(object_id, apc_name))
    obj_path = os.path.join(apc_models_path, '{}.obj'.format(apc_name))
    texture_path = os.path.join(apc_models_path, '{}.png'.format(apc_name))
    ply_path = os.path.join(models_path, 'obj_{:06d}.ply'.format(object_id))
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=1000, axisy=1000, axisz=1000, scalecenter='barycenter')
    ms.apply_filter('compute_texcoord_transfer_wedge_to_vertex')
    ms.save_current_mesh(ply_path, save_vertex_color=False, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)

    with open(ply_path, 'r') as f:
        lines = f.readlines()
    with open(ply_path, 'w') as f:
        for line in lines:
            if line[:11] == 'comment Tex':
                line = 'comment TextureFile obj_{0:06d}.png\n'.format(object_id)
            f.write(line)
    os.system("mv {} {}".format(os.path.join(models_path, "{}.png".format(apc_name)), os.path.join(models_path, "obj_{:06d}.png".format(object_id))))