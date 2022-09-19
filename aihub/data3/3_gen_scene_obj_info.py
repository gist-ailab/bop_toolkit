import json
import os
import glob


output_path = "/home/seung/OccludedObjectDataset/ours/data3/data3_1_raw"


def i2s(num):
    return "{0:06d}".format(num)

scene_id_to_obj_id = {
    "1": "1",
    "2": "11",
    "3": "12",
    "4": "13",
    "5": "14",
    "6": "15",
    "7": "16",
    "8": "17",
    "9": "18",
    "10": "19",
    "11": "21",
    "12": "25",
    "13": "28",
    "14": "31",
    "15": "33",
    "16": "35",
    "17": "38",
    "18": "45",
    "19": "50",
    "20": "51",
    "21": "52",
    "22": "53",
    "23": "54",
    "24": "56",
    "25": "60",
    "26": "61",
    "27": "76",
    "28": "78",
    "29": "80",
    "0": "82",
}


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
        "obj_id": scene_id_to_obj_id[str(scene_id%30)],
        "num_inst": 1,
    }]
    with open(os.path.join(scene_folder_path, "scene_obj_info.json"), 'w') as f:
        json.dump(scene_obj_info, f, indent=2)

