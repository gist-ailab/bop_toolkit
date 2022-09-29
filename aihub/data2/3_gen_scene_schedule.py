import os
import pandas as pd
import numpy as np
import json



if __name__ == '__main__':

    # file name of json to save
    save_file = 'assets/scene_info.json' 
    sch_file = 'assets/scene_info.xlsx'

    sch_data = pd.read_excel(sch_file, engine='openpyxl')

    obj_inst_nums = [int(obj_id.split('object_ID_')[-1]) for obj_id in list(sch_data.columns) if obj_id.startswith('object_ID_')]
    obj_inst_nums = sorted(obj_inst_nums)
    print("NUM-INST:", obj_inst_nums)
    print("---" * 20)
    
    json_data = {}
    for data_idx, scene_id in enumerate(sch_data["scene_number"]):
        # get object ids
        obj_ids = []
        for obj_inst_num in obj_inst_nums:
            obj_id = sch_data["object_ID_{}".format(obj_inst_num)][data_idx]
            if obj_id != obj_id:
                continue
            elif isinstance(obj_id, str) and "(" in obj_id:
                obj_id = int(obj_id.split("(")[-1].split(")")[0])
            elif isinstance(obj_id, float): 
                obj_id = int(obj_id)
            # print("... {} - {}".format(obj_inst_num, obj_id), type(obj_id))
            obj_ids.append(obj_id)
        try: 
            int(scene_id)
        except:
            continue
        # parsing data
        scene_id = str(scene_id)
        json_data[scene_id] = []
        for obj_id in np.unique(obj_ids):
            num_inst = len(np.where(obj_ids==obj_id)[0])
            json_data[scene_id].append({"obj_id": str(obj_id), "num_inst": str(num_inst)})

    print(json_data.keys())

    # save as JSON
    with open(save_file, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
