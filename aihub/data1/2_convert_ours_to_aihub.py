import pandas as pd
import os 
import glob
import json
from tqdm import tqdm

labeling_data_info_path = '/home/seung/Workspace/papers/2022/clora/bop_toolkit/assets/labeling_data_info.xlsx'
input_path = "/home/seung/OccludedObjectDataset/ours/data1"
output_path = '/home/seung/OccludedObjectDataset/aihub/원천데이터/물체3D스캔'

labeling_data_info = pd.read_excel(labeling_data_info_path, engine='openpyxl')

# get labeling data info
super_class_ids = list(labeling_data_info['Unnamed: 6'][1:])
sub_class_ids = list(labeling_data_info['Unnamed: 7'][1:])
semantic_class_ids = list(labeling_data_info['Unnamed: 8'][1:])
object_ids = list(labeling_data_info['Unnamed: 9'][1:])

object_id_to_class_id = {}
for obj_id, super_class_id, sub_class_id, semantic_class_id in zip(object_ids, super_class_ids, sub_class_ids, semantic_class_ids):
    object_id_to_class_id[int(obj_id)] = {
        "super_class": int(super_class_id),
        "sub_class": int(sub_class_id),
        "semantic_class": int(semantic_class_id),
        "object_id": int(obj_id)
    }


for input_obj_path in tqdm(sorted(glob.glob(input_path + "/models_obj/*.obj"))):
    obj_id = int(os.path.basename(input_obj_path).split('_')[-1].split('.')[0])


    input_png_path = os.path.join(input_path, 'models_obj', 'obj_{:06d}.png'.format(obj_id))
    input_mtl_path = os.path.join(input_path, 'models_obj', 'obj_{:06d}.obj.mtl'.format(obj_id))
    input_ply_path = os.path.join(input_path, 'models', 'obj_{:06d}.ply'.format(obj_id))

    if not os.path.exists(input_png_path) or not os.path.exists(input_mtl_path) or not os.path.exists(input_ply_path):
        continue

    metadata = object_id_to_class_id[obj_id]
    output_file_name = 'H1_{}_{}_{}_{}'.format(metadata['super_class'], metadata['sub_class'], metadata['semantic_class'], metadata['object_id'])

    output_obj_path = os.path.join(output_path, 'obj', output_file_name + '.obj')
    output_png_path = os.path.join(output_path, 'obj', output_file_name + '.png')
    output_mtl_path = os.path.join(output_path, 'obj', output_file_name + '.mtl')
    output_ply_path = os.path.join(output_path, 'ply', output_file_name + '.ply')
    output_json_path = os.path.join(output_path, 'json', output_file_name + '.json')


    os.system('cp {} {}'.format(input_obj_path, output_obj_path))
    # os.system('cp {} {}'.format(input_png_path, output_png_path))
    os.system('cp {} {}'.format(input_mtl_path, output_mtl_path))
    os.system('cp {} {}'.format(input_ply_path, output_ply_path))

    # generate json
    json_data = {"object_type": metadata}
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    with open(output_obj_path, 'r') as f:
        lines = f.readlines()
    fout = open(output_obj_path, 'wt')
    for line in lines:
        if 'Object obj_{:06d}.obj'.format(obj_id) in line:
            fout.write(line.replace('obj_{:06d}.obj'.format(obj_id), output_file_name + '.obj'))
        elif 'obj_{:06d}.obj.mtl'.format(obj_id) in line:
            fout.write(line.replace('obj_{:06d}.obj.mtl'.format(obj_id), output_file_name + '.mtl'))
        else:
            fout.write(line)

    fout.close()

    with open(output_mtl_path, 'r') as f:
        lines = f.readlines()
    fout = open(output_mtl_path, 'wt')
    for line in lines:
        if 'obj_{:06d}.png'.format(obj_id) in line:
            fout.write(line.replace('obj_{:06d}.png'.format(obj_id), output_file_name + '.png'))
        else:
            fout.write(line)


