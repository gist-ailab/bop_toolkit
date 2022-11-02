import os
import glob
import pymeshlab
from tqdm import tqdm

ood_root = os.environ['OOD_ROOT']

artec_models_path = os.path.join(ood_root, "ours/data1/artec")
aihub_models_path = os.path.join(ood_root, "ours/data1/models")

for obj_path in tqdm(sorted(glob.glob(artec_models_path + "/*.obj"))):

    id = int(os.path.basename(obj_path).replace(".obj", ""))
    if id != 196:
        continue
    ply_path = os.path.join(aihub_models_path, "obj_{:06d}.ply".format(id))
    print("processing model: obj id: {}".format(id))

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_filter('compute_matrix_by_principal_axis')
    ms.apply_filter('compute_matrix_from_translation', traslmethod = 'Center on Scene BBox')
    ms.apply_filter('compute_color_from_texture_per_vertex')
    ms.save_current_mesh(ply_path, save_vertex_color=True, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False)
    os.system("mv {} {}".format(os.path.join(artec_models_path, "{}_1.png".format(id)), os.path.join(aihub_models_path, "obj_{:06d}.png".format(id))))










