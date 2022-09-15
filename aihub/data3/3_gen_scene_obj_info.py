import json
import os
import glob


output_path = "/home/seung/OccludedObjectDataset/data3/data3_raw"


def i2s(num):
    return "{0:06d}".format(num)



target_scene_numbers = []
for scene_path in sorted(glob.glob(output_path + "/*")):
    scene_number = os.path.basename(scene_path)
    try:
        scene_number = int(scene_number)
        target_scene_numbers.append(scene_number)
    except:
        continue

for scene_id in target_scene_numbers:
    scene_number = i2s(int(scene_id))
    scene_folder_path = os.path.join(output_path, scene_number)
    scene_obj_info = [{
        "obj_id": 13,
        "num_inst": 1,
    }]
    with open(os.path.join(scene_folder_path, "scene_obj_info.json"), 'w') as f:
        json.dump(scene_obj_info, f, indent=2)

