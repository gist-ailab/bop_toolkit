import open3d 
import os

ood_root = os.environ['OOD_ROOT']

# model_1_folder_path = os.path.join(ood_root, "ours/data1/models_notaligned")
model_2_folder_path = os.path.join(ood_root, "ours/data1/models")

for obj_id in range(135, 200):

    print("processing model: obj id: {}".format(obj_id))
    # model_1_path = os.path.join(model_1_folder_path, "obj_{:06d}.ply".format(obj_id))
    model_2_path = os.path.join(model_2_folder_path, "obj_{:06d}.ply".format(obj_id))

    # if not os.path.exists(model_1_path) or not os.path.exists(model_2_path):
        # print("file does not exists. skipping obj id: {}".format(obj_id))
        # continue

    # model_1 = open3d.io.read_point_cloud(model_1_path)
    model_2 = open3d.io.read_point_cloud(model_2_path)

    # open3d.visualization.draw_geometries([model_1, model_2])
    open3d.visualization.draw_geometries([model_2])