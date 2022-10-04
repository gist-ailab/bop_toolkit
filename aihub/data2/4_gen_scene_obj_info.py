import json
import os
import glob


input_json_path = "assets/scene_info.json"
output_path = "/OccludedObjectDataset/ours/data2/data2_real_source/all"


def i2s(num):
    return "{0:06d}".format(num)

with open(input_json_path, 'r') as f:
    scene_obj_infos = json.load(f)

target_scene_numbers = []
for scene_path in sorted(glob.glob(output_path + "/*")):
    scene_number = os.path.basename(scene_path)
    try:
        scene_number = int(scene_number)
        target_scene_numbers.append(scene_number)
    except:
        continue

for scene_id, scene_obj_info in scene_obj_infos.items():
    if int(scene_id) not in target_scene_numbers:
        continue
    scene_number = i2s(int(scene_id))
    scene_folder_path = os.path.join(output_path, scene_number)
    with open(os.path.join(scene_folder_path, "scene_obj_info.json"), 'w') as f:
        json.dump(scene_obj_info, f, indent=2)

