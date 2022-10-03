import pymeshlab
import os
import glob
from tqdm import tqdm

def colored_ply_to_textured_obj(ply_path, obj_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_path)
    ms.apply_filter('compute_texcoord_parametrization_triangle_trivial_per_wedge', textdim=4096, border=1, method=1)
    ms.apply_filter('transfer_attributes_to_texture_per_vertex', attributeenum = 'Vertex Color', textname='{}.png'.format(os.path.basename(ply_path).split('.')[0]), textw=4096, texth=4096)
    ms.save_current_mesh(obj_path, save_vertex_color=False, save_vertex_normal=True, save_face_color=True, save_wedge_texcoord=True)



input_model_path = "/home/seung/OccludedObjectDataset/ours/data1/models_original"
output_model_path = "/home/seung/OccludedObjectDataset/ours/data1/models_obj"

# Attributes to save for the output models.
attrs_to_save = []

# input_obj_paths = ['/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000013.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000067.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000073.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000078.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000082.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000106.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000107.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000108.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000109.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000110.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000111.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000112.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000113.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000114.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000115.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000116.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000117.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000118.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000119.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000120.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000121.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000122.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000123.ply', '/home/seung/OccludedObjectDataset/ours/data1/models_original/obj_000124.ply']


# Process models of all objects in the selected dataset.
error_lists = []
for model_in_path in tqdm(sorted(glob.glob(input_model_path + "/*.ply"))[10:]):
    print("processing: ", model_in_path)
    model_out_path = os.path.join(output_model_path, os.path.basename(model_in_path))
    model_out_path = model_out_path.replace('.ply', '.obj')
    try:
        colored_ply_to_textured_obj(model_in_path, model_out_path)
    except:
        print("error: ", model_in_path)
        error_lists.append(model_in_path)

print("error_lists: ", error_lists)
