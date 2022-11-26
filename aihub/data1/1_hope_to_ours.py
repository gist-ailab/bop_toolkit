
import os
import json 
import pandas as pd
from tqdm import tqdm
import pymeshlab
ood_root = os.environ['OOD_ROOT']

models_info = pd.read_excel("./assets/models_info.ods")

object_ids = []
bop_names = []
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
        if bop_name[:3] == 'obj' and paper == 'HOPE':
            object_ids.append(int(object_id))
            bop_names.append(bop_name)

print('Total number of HOPE objects: {}'.format(len(object_ids)))


for object_id, bop_name in zip(tqdm(object_ids), bop_names):

    print('processing object_id: {}, bop_name: {}'.format(object_id, bop_name))

    ## 1. copy models
    
    # obj
    models_ori = os.path.join(ood_root, 'ours/data1/hope/models/{}.ply'.format(bop_name))
    models_new = os.path.join(ood_root, 'ours/data1/models_notaligned/obj_{:06d}.ply'.format(object_id))
    os.system('cp {} {}'.format(models_ori, models_new))

    # pngs 
    texture_ori = os.path.join(ood_root, 'ours/data1/hope/models/{}.png'.format(bop_name))
    texture_new = os.path.join(ood_root, 'ours/data1/models_notaligned/obj_{:06d}.png'.format(object_id))
    os.system('cp {} {}'.format(texture_ori, texture_new))

    # change texture path in ply
    with open(models_new, 'r') as f:
        lines = f.readlines()
    with open(models_new, 'w') as f:
        for line in lines:
            if line[:11] == 'comment Tex':
                line = 'comment TextureFile obj_{0:06d}.png\n'.format(object_id)
            f.write(line)
    
    # models_info.json
    models_info_ori_path = os.path.join(ood_root, 'ours/data1/hope/models/models_info.json')
    models_info_new_path = os.path.join(ood_root, 'ours/data1/models_notaligned/models_info.json')
    with open(models_info_ori_path, 'r') as f:
        models_info_ori = json.load(f)
    if not os.path.exists(models_info_new_path):
        models_info_new = models_info_ori
    else:
        with open(models_info_new_path, 'r') as f:
            models_info_new = json.load(f)
        if len(models_info_new) == 0:
            models_info_new = models_info_ori
        else:
            models_info_new[str(object_id)] = models_info_ori[str(int(bop_name.split('_')[1]))]
    with open(models_info_new_path, 'w') as f:
        json.dump(models_info_new, f, indent=4)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(models_new)
    ms.apply_filter('compute_matrix_from_rotation', rotaxis = 'X axis', angle=-90)
    ms.apply_filter('compute_matrix_from_rotation', rotaxis = 'Y axis', angle=-90)
    ms.save_current_mesh(models_new, save_vertex_color=False, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)



    # 2. copy models_eval

    # # obj
    # models_eval_ori = os.path.join(ood_root, 'ours/data1/hope/models_eval/{}.ply'.format(bop_name))
    # models_eval_new = os.path.join(ood_root, 'ours/data1/models_eval/obj_{:06d}.ply'.format(object_id))
    # os.system('cp {} {}'.format(models_eval_ori, models_eval_new))

    # # models_info.json
    # models_info_ori_path = os.path.join(ood_root, 'ours/data1/hope/models_eval/models_info.json')
    # models_info_new_path = os.path.join(ood_root, 'ours/data1/models_eval/models_info.json')
    # with open(models_info_ori_path, 'r') as f:
    #     models_info_ori = json.load(f)
    # if not os.path.exists(models_info_new_path):
    #     models_info_new = models_info_ori
    # else:
    #     with open(models_info_new_path, 'r') as f:
    #         models_info_new = json.load(f)
    #     if len(models_info_new) == 0:
    #         models_info_new = models_info_ori
    #     else:
    #         models_info_new[str(object_id)] = models_info_ori[str(int(bop_name.split('_')[1]))]
    # with open(models_info_new_path, 'w') as f:
    #     json.dump(models_info_new, f, indent=4)