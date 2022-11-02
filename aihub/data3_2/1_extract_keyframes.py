import os
import glob
import json
import sys
import rospy
import shutil

ood_root = os.environ['OOD_ROOT']
dataset_root = os.path.join(ood_root, 'ours/data3/data3_2_raw/')
output_root = os.path.join(ood_root, 'ours/data3/data3_2_source/all')

ignore_secs_i = 2
ignore_secs_f = 2

scene_ids = sorted([int(x) for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])
for scene_id in scene_ids:

    scene_folder_path = os.path.join(dataset_root, "{0:06d}".format(scene_id))
    with open(os.path.join(scene_folder_path, "scene_camera.json"), 'r') as j_file:
        scene_camera_info = json.load(j_file)
    with open(os.path.join(scene_folder_path, "robot_info.json"), 'r') as j_file:
        robot_info = json.load(j_file)
    with open(os.path.join(scene_folder_path, "user_command.json"), 'r') as j_file:
        user_command = json.load(j_file)

    # get ros time stamps for each image
    im_timestamps = []
    im_ids = scene_camera_info.keys()
    im_ids_cam1 = [x for x in im_ids if int(x) % 4 == 1 ]
    for im_id_cam1 in im_ids_cam1:
        secs = scene_camera_info[im_id_cam1]['secs']
        nsecs = scene_camera_info[im_id_cam1]['nsecs']
        im_timestamps.append(rospy.Time(secs, nsecs))
    
    first_secs = im_timestamps[0].secs
    last_secs = im_timestamps[-1].secs
    selected_im_timestamps = []
    selected_im_ids = []
    # crop the first and last few seconds
    for im_timestamp, im_id in zip(im_timestamps, im_ids):
        if im_timestamp.secs >= first_secs + ignore_secs_i and im_timestamp.secs <= last_secs - ignore_secs_f:
            selected_im_timestamps.append(im_timestamp)
            selected_im_ids.append(im_id)
    
    # select only middle frames for each second
    selected_im_timestamps_2 = []
    selected_im_ids_2 = []
    for i in range(first_secs + ignore_secs_i, last_secs - ignore_secs_f):
        im_timestamps_i = [x for x in selected_im_timestamps if x.secs == i]
        im_ids_i = [x for x, y in zip(selected_im_ids, selected_im_timestamps) if y.secs == i]
        if len(im_timestamps_i) > 0:
            selected_im_timestamps_2.append(im_timestamps_i[len(im_timestamps_i)//2])
            selected_im_ids_2.append(im_ids_i[len(im_timestamps_i)//2])
    
    selected_im_ids_cam1 = selected_im_ids_2
    selected_im_timestamps =  selected_im_timestamps_2
    
    # get ros time stamps for each robot state
    robot_timestamps = []
    for robot_state in robot_info:
        secs = robot_state['secs']
        nsecs = robot_state['nsecs']
        robot_timestamps.append(rospy.Time(secs, nsecs))

    # get ros time stamps for each user command
    user_timestamps = []
    for user_state in user_command:
        secs = user_state['secs']
        nsecs = user_state['nsecs']
        user_timestamps.append(rospy.Time(secs, nsecs))

    # get closest index of robot state for each image
    robot_state_indices = []
    for image_timestamp in selected_im_timestamps:
        robot_state_indices.append(min(range(len(robot_timestamps)), key=lambda i: abs(robot_timestamps[i] - image_timestamp)))

    # get closest index of user command for each image
    user_command_indices = []
    for image_timestamp in selected_im_timestamps:
        user_command_indices.append(min(range(len(user_timestamps)), key=lambda i: abs(user_timestamps[i] - image_timestamp)))

    # log selected frames and indices
    selected_info = {}
    selected_info['im_ids'] = selected_im_ids_cam1
    selected_info['robot_state_indices'] = robot_state_indices
    selected_info['user_command_indices'] = user_command_indices
    with open(os.path.join(output_root, "{0:06d}".format(scene_id), "selected_info.json"), 'w') as j_file:
        json.dump(selected_info, j_file, indent=4)

    # move to output folder
    new_camera_info = {}
    new_robot_info = []
    new_user_command = []
    for new_glob_im_id, (im_id_cam1, robot_state_index, user_command_index) in enumerate(zip(selected_im_ids_cam1, robot_state_indices, user_command_indices)):
        for idx in range(4):
            old_im_id = int(im_id_cam1) + idx
            new_im_id = new_glob_im_id * 4 + idx + 1
            new_scene_folder_path = os.path.join(output_root, "{0:06d}".format(scene_id))
            old_rgb_path = os.path.join(scene_folder_path, "rgb", "{0:06d}.png".format(old_im_id))
            old_depth_path = os.path.join(scene_folder_path, "depth", "{0:06d}.png".format(old_im_id))
            new_rgb_path = os.path.join(new_scene_folder_path, "rgb", "{0:06d}.png".format(new_im_id))
            new_depth_path = os.path.join(new_scene_folder_path, "depth", "{0:06d}.png".format(new_im_id))
            os.makedirs(new_scene_folder_path, exist_ok=True)
            os.makedirs(os.path.join(new_scene_folder_path, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(new_scene_folder_path, "depth"), exist_ok=True)
            shutil.copy(old_rgb_path, new_rgb_path)
            shutil.copy(old_depth_path, new_depth_path)
            new_camera_info[str(new_im_id)] = scene_camera_info[str(old_im_id)]
        new_robot_info.append(robot_info[robot_state_index])
        new_user_command.append(user_command[user_command_index])
    
    # save new camera info
    with open(os.path.join(output_root, "{0:06d}".format(scene_id), "camera_info.json"), 'w') as j_file:
        json.dump(new_camera_info, j_file, indent=4)
    with open(os.path.join(output_root, "{0:06d}".format(scene_id), "robot_info.json"), 'w') as j_file:
        json.dump(new_robot_info, j_file, indent=4)
    with open(os.path.join(output_root, "{0:06d}".format(scene_id), "user_command.json"), 'w') as j_file:
        json.dump(new_user_command, j_file, indent=4)


    print(robot_state_indices)
    exit()


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

    

