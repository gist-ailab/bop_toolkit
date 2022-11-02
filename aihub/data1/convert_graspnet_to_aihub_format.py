import os
import glob
import pymeshlab
from tqdm import tqdm


graspnet_models_path = "/home/seung/OccludedObjectDataset/ours/data1/graspnet_billion"
aihub_models_path = "/home/seung/OccludedObjectDataset/ours/data1/models_original"

graspnet_to_aihub_id = {
    "045": "115"
    # "042": "104",
    # "047": "105",
    # "050": "106",
    # "053": "107",
    # "054": "108",
    # "067": "109",
    # "060": "110",
    # "062": "111",
    # "063": "112",
    # "064": "113",
    # "065": "114",
    # "066": "115",
    # "070": "116",
    # "074": "117",
    # "033": "118",
    # "037": "119",
    # "059": "120",
    # "039": "180",
    # "059": "181",
    # # dexnet
    # "075": "121",
    # "076": "122",
    # "077": "123",
    # "078": "124",
    # "079": "125",
    # "080": "126",
    # "081": "127",
    # "082": "128",
    # "083": "129",
    # "084": "130",
    # "085": "131",
    # "086": "132",
    # "087": "133",
}


for graspnet_model_id in tqdm(graspnet_to_aihub_id):
    print("processing model: {}".format(graspnet_model_id))

    obj_path = os.path.join(graspnet_models_path, graspnet_model_id, "textured.obj")
    ply_path = os.path.join(aihub_models_path, "obj_{:06d}.ply".format(int(graspnet_to_aihub_id[graspnet_model_id])))

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=1000, axisy=1000, axisz=1000, scalecenter='barycenter')
    ms.apply_filter('compute_color_from_texture_per_vertex')
    ms.save_current_mesh(ply_path, save_vertex_color=True, save_vertex_normal=True, save_face_color=False, save_wedge_texcoord=False)
    os.system("mv {} {}".format(os.path.join(aihub_models_path, "textured.jpg"), os.path.join(aihub_models_path, "obj_{:06d}.jpg".format(int(graspnet_to_aihub_id[graspnet_model_id])))))











