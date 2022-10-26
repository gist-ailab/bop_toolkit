
import os
import json 
import pandas as pd
from tqdm import tqdm
import numpy as np
ood_root = os.environ['OOD_ROOT']

models_info = pd.read_excel("./assets/models_info.ods")

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
    obj_ori_path = os.path.join(ood_root, 'ours/data1/ycb/{}/google_16k/textured.obj'.format(ycb_name))
    mtl_ori_path = os.path.join(ood_root, 'ours/data1/ycb/{}/google_16k/textured.mtl'.format(ycb_name))
    texture_ori_path = os.path.join(ood_root, 'ours/data1/ycb/{}/google_16k/texture_map.png'.format(ycb_name))
    obj_new_path = os.path.join(ood_root, 'ours/data1/models_obj/obj_{:06d}.obj'.format(object_id))
    mtl_new_path = os.path.join(ood_root, 'ours/data1/models_obj/obj_{:06d}.mtl'.format(object_id))
    texture_new_path = os.path.join(ood_root, 'ours/data1/models_obj/obj_{:06d}.png'.format(object_id))
    os.system('cp {} {}'.format(obj_ori_path, obj_new_path))
    os.system('cp {} {}'.format(mtl_ori_path, mtl_new_path))
    os.system('cp {} {}'.format(texture_ori_path, texture_new_path))

    # change mtl name of obj file
    with open(model_out_path, 'r') as f:
        lines = f.readlines()
    with open(model_out_path, 'w') as f:
        for line in lines:
            if line.startswith('mtllib'):
                line = line.replace('.obj.mtl', '.mtl')
            f.write(line)
    
    # change mtl file name
    mtl_in_path = model_out_path.replace('.obj', '.obj.mtl')
    mtl_out_path = mtl_in_path.replace('.obj.mtl', '.mtl')
    os.rename(mtl_in_path, mtl_out_path)

    # add texture info to mtl file
    with open(model_out_path.replace('.obj', '.mtl'), 'r') as f:
        lines = f.readlines()
    with open(model_out_path.replace('.obj', '.mtl'), 'w') as f:
        for idx, line in enumerate(lines):
            if idx == len(lines) - 1:
                line = 'map_Kd obj_{:06d}.png'.format(object_id)
            f.write(line)

    #!TODO: scal, centerize?, exports as ply