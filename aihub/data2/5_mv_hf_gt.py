import glob
import os
import shutil
from tqdm import tqdm

ood_root = os.environ['OOD_ROOT']
hf_gt_folder_path = os.path.join(ood_root, 'ours/data2/hf_gt/Data2')
dataset_root = os.path.join(ood_root, 'ours/data2/data2_real_source/all')

for file_path in tqdm(glob.iglob(hf_gt_folder_path + "/**", recursive=True)):
    # iterate over all folders 
    if 'scene_gt' in file_path:
        scene_id = os.path.basename(file_path).split("_")[2].split(".")[0]
        scene_folder_path = os.path.join(dataset_root, "{0:06d}".format(int(scene_id)))
        if not os.path.isdir(scene_folder_path):
            print(scene_folder_path)
            continue
        if not os.path.exists(os.path.join(scene_folder_path, "scene_gt_{0:06d}.json".format(int(scene_id)))):
            shutil.copy(file_path, scene_folder_path)
        if int(scene_id) in [484, 501, 509, 650, 564, 304, 305, 842, 1043]:
            shutil.copy(file_path, scene_folder_path)




# check if the scene_gt exists
scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
n_annotated_folders = 0
for scene_id in scene_ids:
    if not os.path.exists(os.path.join(dataset_root, "{0:06d}".format(scene_id), "scene_gt_{0:06d}.json".format(scene_id))):
        print("scene_gt.json does not exist in {}".format(os.path.join(dataset_root, "{0:06d}".format(scene_id))))
    else:
        n_annotated_folders += 1 
print("Total {} folders are annotated".format(n_annotated_folders))
    