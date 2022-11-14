
import os
import json 
import pandas as pd
from tqdm import tqdm
import numpy as np
import pymeshlab


ood_root = os.environ['OOD_ROOT']
models_info = pd.read_excel("./assets/models_info.ods")
models_path = os.path.join(ood_root, "ours/data1/models")

object_ids = []
ycb_names = []
for object_id, bop_name, ycb_name, paper in zip(models_info["object_id"], models_info["model BOP"], models_info["original name"], models_info['논문']):
    try: 
        int(object_id)
    except:
        continue
    else:
        if not isinstance(bop_name, str) and not isinstance(bop_name, float):
            continue
        if isinstance(bop_name, float):
            if np.isnan(bop_name):
                bop_name = str(bop_name)
            else:
                continue
        if not isinstance(paper, str):
            continue
        if bop_name[:3] != 'obj' and paper == 'YCB':
            object_ids.append(int(object_id))
            ycb_names.append(ycb_name)


print('Total number of YCB objects: {}'.format(len(object_ids)))
for object_id, ycb_name in zip(tqdm(object_ids), ycb_names):

    print('processing object_id: {}, ycb_name: {}'.format(object_id, ycb_name))

    # copy obj files
    obj_path = os.path.join(ood_root, 'ours/data1/ycb/{}/google_16k/textured.obj'.format(ycb_name))
    texture_path = os.path.join(ood_root, 'ours/data1/ycb/{}/google_16k/texture_map.png'.format(ycb_name))
    ply_path = os.path.join(models_path, 'obj_{:06d}.ply'.format(object_id))
    # mtl_ori_path = os.path.join(ood_root, 'ours/data1/ycb/{}/google_16k/textured.mtl'.format(ycb_name))
    # texture_ori_path = os.path.join(ood_root, 'ours/data1/ycb/{}/google_16k/texture_map.png'.format(ycb_name))
    # obj_new_path = os.path.join(ood_root, 'ours/data1/models_obj/obj_{:06d}.obj'.format(object_id))
    # mtl_new_path = os.path.join(ood_root, 'ours/data1/models_obj/obj_{:06d}.mtl'.format(object_id))
    # texture_new_path = os.path.join(ood_root, 'ours/data1/models_obj/obj_{:06d}.png'.format(object_id))
    # os.system('cp {} {}'.format(obj_ori_path, obj_new_path))
    # os.system('cp {} {}'.format(mtl_ori_path, mtl_new_path))
    # os.system('cp {} {}'.format(texture_ori_path, texture_new_path))

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
    os.system("cp {} {}".format(texture_path, os.path.join(models_path, "obj_{:06d}.png".format(object_id))))