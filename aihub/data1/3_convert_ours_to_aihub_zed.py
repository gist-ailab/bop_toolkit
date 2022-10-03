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


for img_folder_path in tqdm(sorted(glob.glob(input_path + "/ZED/*"))):

    obj_id = int(os.path.basename(img_folder_path))
    rgb_folder_path = os.path.join(img_folder_path, 'rgb')
    depth_folder_path = os.path.join(img_folder_path, 'depth')
    metadata = object_id_to_class_id[obj_id]

    img_ids = sorted([int(x.split('.')[0].split('_')[1]) for x in os.listdir(rgb_folder_path) if os.path.isfile(os.path.join(rgb_folder_path, x))])

    for img_id in img_ids:
        output_file_name = 'H1_{}_{}_{}_{}_{}'.format(metadata['super_class'], metadata['sub_class'], metadata['semantic_class'], metadata['object_id'], img_id)
        input_rgb_path = os.path.join(rgb_folder_path, 'rgb_{}.png'.format(img_id))
        input_depth_path = os.path.join(depth_folder_path, 'depth_im_{}.png'.format(img_id))
        output_rgb_path = os.path.join(output_path, 'rgb', output_file_name + '.png')
        output_depth_path = os.path.join(output_path, 'depth', output_file_name + '.png')

        os.system('cp {} {}'.format(input_rgb_path, output_rgb_path))
        os.system('cp {} {}'.format(input_depth_path, output_depth_path))
