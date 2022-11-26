import os
import glob
import pymeshlab


ood_root = os.environ['OOD_ROOT']
input_model_path = os.path.join(ood_root, "ours/data1/models")
output_model_path = os.path.join(ood_root, "ours/data1/models_obj")

# Process models of all objects in the selected dataset.
for model_in_path in glob.glob(input_model_path + "/*.ply"):
  obj_id = os.path.basename(model_in_path).split("_")[-1].split(".")[0]
  print("Processing", model_in_path)
  model_out_path = os.path.join(output_model_path, os.path.basename(model_in_path).replace(".ply", ".obj"))
  texture_out_path = os.path.join(output_model_path, os.path.basename(model_in_path).replace(".ply", ".png"))
  mtl_out_path = os.path.join(output_model_path, os.path.basename(model_in_path).replace(".ply", ".obj.mtl"))
  ms = pymeshlab.MeshSet()
  ms.load_new_mesh(model_in_path)
  ms.save_current_mesh(model_out_path)

  # change the texture path in the mtl file
  if os.path.exists(mtl_out_path) and os.path.exists(texture_out_path):
    with open(mtl_out_path, 'r') as f:
      lines = f.readlines()
    with open(mtl_out_path, 'w') as f:
      for idx, line in enumerate(lines):
        if idx == 6:
          f.write('map_Kd obj_{0:06d}.png\n'.format(int(obj_id)))
        f.write(line)
