import glob
import os
import shutil

hf_gt_folder_path = '/home/seung/OccludedObjectDataset/ours/data2/hf_gt'
dataset_root = "/home/seung/OccludedObjectDataset/ours/data2/data2_real_source/all"


for hf_gt_folder in glob.glob(hf_gt_folder_path + "/*"):

    print("processing {}".format(hf_gt_folder))

    for gt_json in glob.glob(hf_gt_folder + "/scene_gt*.json"):
        scene_id = os.path.basename(gt_json).split("_")[2].split(".")[0]
        print("scene_id: {}".format(scene_id))
        output_scene_dir = os.path.join(dataset_root, scene_id)
        shutil.copy(gt_json, output_scene_dir)

    inner_folders = glob.glob(hf_gt_folder + "/*")
    for inner_folder in inner_folders:
        for gt_json in glob.glob(inner_folder + "/scene_gt*.json"):
            scene_id = os.path.basename(gt_json).split("_")[2].split(".")[0]
            print("scene_id: {}".format(scene_id))
            output_scene_dir = os.path.join(dataset_root, scene_id)
            shutil.copy(gt_json, output_scene_dir)

# check if the scene_gt exists
scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
for scene_id in scene_ids:
    if not os.path.exists(os.path.join(dataset_root, "{0:06d}".format(scene_id), "scene_gt_{0:06d}.json".format(scene_id))):
        print("scene_gt.json does not exist in {}".format(os.path.join(dataset_root, "{0:06d}".format(scene_id))))