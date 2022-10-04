import os
import glob
import json

dataset_root = "/home/ailab/OccludedObjectDataset/ours/data3/data3_1_real_raw"
output_root = "/home/ailab/OccludedObjectDataset/ours/data3/data3_1_real_source"

scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
scene_ids = [x for x in scene_ids if int(x) > 60]
for scene_id in scene_ids:

    scene_folder_path = os.path.join(dataset_root, "{0:06d}".format(scene_id))
    with open(os.path.join(scene_folder_path, "scene_camera.json"), 'r') as j_file:
        scene_camera_info = json.load(j_file)
    
    # get secs for each images
    secs = []
    nsecs = []
    im_ids = []
    im_ids_all = []
    for im_id in scene_camera_info.keys():
        im_ids_all.append(im_id)
        if int(im_id) % 3 != 1:
            continue # only use the first camera
        secs.append(scene_camera_info[im_id]["secs"])
        nsecs.append(scene_camera_info[im_id]["nsecs"])
        im_ids.append(im_id)
    unique_secs = list(set(secs))
    unique_secs.sort()
    if len(unique_secs) < 15:
        print("Not enough secs ({}) in scene {}".format(len(unique_secs), scene_id))
        break
    # get 15 seconds
    unique_secs = unique_secs[:15]
    print(len(unique_secs))

    # for each unique secs, get the im_id of median nsecs
    im_ids_to_use = []
    for unique_sec in unique_secs:
        im_ids_in_sec = [im_id for im_id, sec in zip(im_ids, secs) if sec == unique_sec]
        nsecs_in_sec = [nsec for nsec, sec in zip(nsecs, secs) if sec == unique_sec]
        median_nsecs = sorted(nsecs_in_sec)[len(nsecs_in_sec) // 2]
        im_ids_to_use.append(im_ids_in_sec[nsecs_in_sec.index(median_nsecs)])

    # im_ids_to_use = sorted(im_ids_to_use)[len(im_ids_to_use) // 2 - 7 : len(im_ids_to_use) // 2 + 8]
    
    # add im_ids of second and third camras
    im_ids_to_use = ["{0:06d}".format(int(im_id) + i) for im_id in im_ids_to_use for i in range(3)]
    im_ids_to_use = sorted(im_ids_to_use, key=lambda x: int(x))

    # get the secs of minimum and maximum im_ids
    min_sec = scene_camera_info[str(int(im_ids_to_use[0]))]["secs"]
    max_sec = scene_camera_info[str(int(im_ids_to_use[-1]))]["secs"]
    print("Scene {} - {} - {} = {}".format(scene_id, min_sec, max_sec, max_sec - min_sec))

    # get all im_ids between the first and last seconds
    im_ids_to_use_all = ["{0:06d}".format(int(im_id)) for im_id in im_ids_all if min_sec <= scene_camera_info[str(int(im_id))]["secs"] <= max_sec]
    im_ids_to_use_all = sorted(im_ids_to_use_all, key=lambda x: int(x))

    # set new ids starting from 1
    first_id = int(im_ids_to_use_all[0])
    new_im_ids_to_use = ["{0:06d}".format(int(im_id) - first_id + 1) for im_id in im_ids_to_use]
    new_im_ids_to_use_all = ["{0:06d}".format(int(im_id) - first_id + 1) for im_id in im_ids_to_use_all]

    scene_folder_path_new = os.path.join(output_root, "{0:06d}".format(scene_id))
    os.makedirs(scene_folder_path_new, exist_ok=True)
    os.makedirs(os.path.join(scene_folder_path_new, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(scene_folder_path_new, "depth"), exist_ok=True)

    # copy images to the new folder
    # for im_id in im_ids_to_use_all:
    #     im_path = os.path.join(scene_folder_path, "rgb", "{0:06d}.png".format(int(im_id)))
    #     im_path_new = os.path.join(scene_folder_path_new, "rgb", "{0:06d}.png".format(int(im_id) - first_id + 1))
    #     os.system("cp {} {}".format(im_path, im_path_new))
    # for im_id in im_ids_to_use_all:
    #     im_path = os.path.join(scene_folder_path, "depth", "{0:06d}.png".format(int(im_id)))
    #     im_path_new = os.path.join(scene_folder_path_new, "depth", "{0:06d}.png".format(int(im_id) - first_id + 1))
    #     os.system("cp {} {}".format(im_path, im_path_new))
    # # copy scene_camera.json to the new folder
    # with open(os.path.join(scene_folder_path_new, "scene_camera.json"), 'w') as j_file:
    #     # get the camera info of the selected images
    #     scene_camera_info_new = {str(int(im_id)-first_id+1): scene_camera_info[str(int(im_id))] for im_id in im_ids_to_use_all}
    #     json.dump(scene_camera_info_new, j_file, indent=4)
    
    # extract only the robot_info of selected images
    with open(os.path.join(scene_folder_path, "robot_info.json"), 'r') as j_file:
        robot_info = json.load(j_file)
        robot_info_new = robot_info
        for key in ["robot_joint_position", "robot_joint_velocity", "gripper_joint_position", "gripper_joint_velocity", "command_value"]:
            if len(robot_info_new[key]) == 1:
                robot_info_new[key] = [0.0] * (len(im_ids_to_use_all) // 3)
            else:
                robot_info_new[key] = [robot_info[key][(int(im_id) -1 ) // 3-1] for im_id in im_ids_to_use_all]
    with open(os.path.join(scene_folder_path_new, "robot_info.json"), 'w') as j_file:
        json.dump(robot_info_new, j_file, indent=4)

    # save keyframe ids
    with open(os.path.join(scene_folder_path_new, "keyframes.json"), 'w') as f:
        json.dump(new_im_ids_to_use, f, indent=4)
        
    print("Scene {} done".format(scene_id))
print()

    

