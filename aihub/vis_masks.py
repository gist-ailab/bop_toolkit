import os
import glob
import cv2
import imgviz
import matplotlib.pyplot as plt
import numpy as np
scene_id = 0
img_id = 0
dataset_path = "/home/seung/OccludedObjectDataset/ours/data2/data2_syn_source/train_pbr"
# dataset_path = "/home/seung/OccludedObjectDataset/ours/data2/data2_real_source/all"

scene_path = os.path.join(dataset_path, "{:06d}".format(scene_id))
im_path = os.path.join(scene_path, "rgb/{:06d}.jpg".format(img_id))
mask_paths = glob.glob(scene_path + "/mask_visib/{:06d}_*.png".format(img_id))
im = cv2.imread(im_path)
masks = [np.array(cv2.imread(mask_path)[:, :, 0], dtype=bool) for mask_path in mask_paths]

maskviz = imgviz.instances2rgb(im, masks=masks, labels=[int(x) for x in range(len(masks))], line_width=0, alpha=0.3)

plt.figure(dpi=400)
plt.imshow(maskviz)
plt.axis("off")


img = imgviz.io.pyplot_to_numpy()
cv2.imwrite("/home/seung/img.png", img)
plt.close()