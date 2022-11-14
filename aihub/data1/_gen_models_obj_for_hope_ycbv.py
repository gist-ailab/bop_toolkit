import pymeshlab
import os
import glob
from tqdm import tqdm
import os
import json 
import pandas as pd
from tqdm import tqdm


def colored_ply_to_textured_obj(ply_path, obj_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_path)
    # ms.apply_filter('compute_texcoord_parametrization_triangle_trivial_per_wedge', textdim=4096, border=1, method=1)
    # ms.apply_filter('transfer_attributes_to_texture_per_vertex', attributeenum = 'Vertex Color', textname='{}.png'.format(os.path.basename(ply_path).split('.')[0]), textw=4096, texth=4096)
    ms.save_current_mesh(obj_path)


ood_root = os.environ['OOD_ROOT']
models_info = pd.read_excel("./assets/models_info.ods")

object_ids = []
for object_id, bop_name, paper in zip(models_info["object_id"], models_info["model BOP"], models_info['논문']):
    try: 
        int(object_id)
    except:
        continue
    else:
        if not isinstance(bop_name, str):
            continue
        if not isinstance(paper, str):
            continue
        if bop_name[:3] == 'obj' and paper in ['YCB', 'HOPE'] :
            object_ids.append(int(object_id))



for object_id in tqdm(object_ids):
    model_in_path = os.path.join(ood_root, 'ours/data1/models/obj_{:06d}.ply'.format(object_id))
    print("processing: ", model_in_path)
    model_out_path = model_in_path.replace('.ply', '.obj').replace('models', 'models_obj')
    colored_ply_to_textured_obj(model_in_path, model_out_path)

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
