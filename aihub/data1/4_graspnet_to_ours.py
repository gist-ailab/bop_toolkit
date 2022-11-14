import os
import glob
import pymeshlab
from tqdm import tqdm
import cv2
import os
import pandas as pd
from tqdm import tqdm
ood_root = os.environ['OOD_ROOT']

models_info = pd.read_excel("./assets/models_info.ods")

graspnet_models_path = os.path.join(ood_root, "ours/data1/graspnet_billion")
models_path = os.path.join(ood_root, "ours/data1/models")

graspnet_to_aihub_id = {
    "045": "115",
    "042": "104",
    "047": "105",
    "050": "106",
    "053": "107",
    "054": "108",
    "067": "109",
    "060": "110",
    "062": "111",
    "063": "112",
    "064": "113",
    "065": "114",
    "066": "115",
    "070": "116",
    "074": "117",
    "033": "118",
    "037": "119",
    "059": "120",
    "039": "180",
    "059": "181",
    # dexnet
    "075": "121",
    "076": "122",
    "077": "123",
    "078": "124",
    "079": "125",
    "080": "126",
    "081": "127",
    "082": "128",
    "083": "129",
    "084": "130",
    "085": "131",
    "086": "132",
    "087": "133",
}


for graspnet_model_id in tqdm(graspnet_to_aihub_id):
    object_id = int(graspnet_to_aihub_id[graspnet_model_id])
    print("processing model: {} -> obj_id: {}".format(graspnet_model_id, object_id))

    obj_path = os.path.join(graspnet_models_path, graspnet_model_id, "textured.obj")
    ply_path = os.path.join(models_path, "obj_{:06d}.ply".format(object_id))
    if os.path.exists(os.path.join(graspnet_models_path, graspnet_model_id, "textured.jpg")):
        texture_path = os.path.join(graspnet_models_path, graspnet_model_id, "textured.jpg")
    elif os.path.exists(os.path.join(graspnet_models_path, graspnet_model_id, "textured.png")):
        texture_path = os.path.join(graspnet_models_path, graspnet_model_id, "textured.png")
    else:
        texture_path = None
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=1000, axisy=1000, axisz=1000, scalecenter='barycenter')

    if texture_path is not None:
        ms.apply_filter('compute_texcoord_transfer_wedge_to_vertex')
        ms.save_current_mesh(ply_path, save_vertex_color=False, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        with open(ply_path, 'w') as f:
            for line in lines:
                if line[:11] == 'comment Tex':
                    line = 'comment TextureFile obj_{0:06d}.png\n'.format(object_id)
                f.write(line)
        os.system("cp {} {}".format(texture_path, os.path.join(models_path, "obj_{:06d}.png".format(object_id))))
            
    else:
        ms.save_current_mesh(ply_path, save_vertex_color=True, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False, binary=False)

  
    











