import os
import glob
import json

dataset_root = "/home/seung/OccludedObjectDataset/ours/data3/data3_1_raw"

scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
for scene_id in scene_ids:

    scene_folder_path = os.path.join(dataset_root, "{0:06d}".format(scene_id))
    with open(os.path.join(scene_folder_path, "scene_camera.json"), 'r') as j_file:
        scene_camera_info = json.load(j_file)
    
    # get secs for each images
    secs = []
    nsecs = []
    im_ids = []
    for im_id in scene_camera_info.keys():
        if int(im_id) % 3 != 1:
            continue # only use the first camera
        secs.append(scene_camera_info[im_id]["secs"])
        nsecs.append(scene_camera_info[im_id]["nsecs"])
        im_ids.append(im_id)
    unique_secs = list(set(secs))
    unique_secs.sort()
    # for each unique secs, get the im_id of median nsecs
    im_ids_to_use = []
    for unique_sec in unique_secs:
        im_ids_in_sec = [im_id for im_id, sec in zip(im_ids, secs) if sec == unique_sec]
        nsecs_in_sec = [nsec for nsec, sec in zip(nsecs, secs) if sec == unique_sec]
        median_nsecs = sorted(nsecs_in_sec)[len(nsecs_in_sec) // 2]
        im_ids_to_use.append(im_ids_in_sec[nsecs_in_sec.index(median_nsecs)])
    # select the median 15 images
    if len(im_ids_to_use) < 15:
        print("Not enough secs ({}) in scene {}".format(len(im_ids_to_use), scene_id))
    im_ids_to_use = sorted(im_ids_to_use)[len(im_ids_to_use) // 2 - 7 : len(im_ids_to_use) // 2 + 8]
    
    # add im_ids of second and third camras
    im_ids_to_use = ["{0:06d}".format(int(im_id) + i) for im_id in im_ids_to_use for i in range(3)]
    im_ids_to_use = sorted(im_ids_to_use, key=lambda x: int(x))
    

    # save the image ids to use as json
    with open(os.path.join(scene_folder_path, "keyframes.json"), 'w') as j_file:
        json.dump(im_ids_to_use, j_file)
    print("Scene {} done".format(scene_id))
print("Done")

    

