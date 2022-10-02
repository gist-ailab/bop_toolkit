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
