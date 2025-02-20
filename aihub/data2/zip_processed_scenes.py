import os 
import json
import zipfile


with open(os.path.join("./assets/processed_scene_ids.json"), 'r') as j_file:
    processed_scene_ids = json.load(j_file)
processed_scene_ids = [440, 497,  680, 718, 760, 836, 888, 906, 907, 912, 976, 973, 977, 544, 597, 598]


zipped_file_name = "20221108_data2_real_source.zip"
# zip already processed scenes
ood_root = os.environ['OOD_ROOT']

source_paths = []
for scene_id in processed_scene_ids:
    dataset_root = os.path.join(ood_root, 'ours/data2/data2_real_source/all')
    scene_path = os.path.join(dataset_root, "{:06d}".format(scene_id))
    source_paths.append(scene_path)

source_paths = " ".join(source_paths)
cmd = "zip -r {} {}".format(zipped_file_name, source_paths)
os.system(cmd)