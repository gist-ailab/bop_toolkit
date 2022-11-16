import os
import glob
import pymeshlab
from tqdm import tqdm

ood_root = os.environ['OOD_ROOT']

artec_models_path = os.path.join(ood_root, "ours/data1/artec")
aihub_models_path = os.path.join(ood_root, "ours/data1/models_notaligned")

for obj_path in tqdm(sorted(glob.glob(artec_models_path + "/*.obj"))):

    obj_id = int(os.path.basename(obj_path).replace(".obj", ""))
    if obj_id != 83:
        continue
    ply_path = os.path.join(aihub_models_path, "obj_{:06d}.ply".format(obj_id))
    print("processing model: obj_id: {}".format(obj_id))

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_filter('compute_matrix_by_principal_axis')
    ms.apply_filter('compute_matrix_from_translation', traslmethod = 'Center on Scene BBox')
    ms.apply_filter('compute_texcoord_transfer_wedge_to_vertex')
    ms.save_current_mesh(ply_path, save_vertex_color=False, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)

    with open(ply_path, 'r') as f:
        lines = f.readlines()
    with open(ply_path, 'w') as f:
        for line in lines:
            if line[:11] == 'comment Tex':
                line = 'comment TextureFile obj_{0:06d}.png\n'.format(obj_id)
            f.write(line)

    os.system("mv {} {}".format(os.path.join(aihub_models_path, "{}_1.png".format(obj_id)), os.path.join(aihub_models_path, "obj_{:06d}.png".format(obj_id))))










