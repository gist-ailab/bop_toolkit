import os


scene_ids =  [35, 36, 37, 38, 39, 40]
for scene_id in scene_ids:
    print("rsync -rv {:06d}/*.json ailab@172.27.183.190:~/OccludedObjectDataset/ours/data2/data2_real_source/all/{:06d}/".format(scene_id, scene_id))