# !TODO:


## Folder structure
```
# Public objects
ycbv: 3d object models of YCB-Video dataset
hope (28) : 3d object models of HOPE dataset
hb: 3d object models of homebrewedDB dataset
apc: 3d object models of amazon bin picking challenge
adversarial: adverarial objects of DexNet
graspnet_billion: 3d object models of Grasp1Billion dataset


# Ours
artec: 3d object models scanned with Artec LEO by GIST AILAB (108 objects)
models_anno: ply (colored points) for 6d object pose annotation
models_aligned: models (only artec) aligned with z-axis 
models: ply (colored points + texture) compatible with bop's models 
- with texture -> x, y, z, nx, ny , nz, texture_u, texture_v , face, vertex_indices

models_eval: ply (colored points + texture) compatible with bop's models_eval
models_obj: obj (colored points + texture) for aihub submission 
```


- [x] HOPE -> models, models_eval
- [ ] YCBV -> models, models_eval
- [ ] YCB  -> models, models_eval
- [ ] models -> models_anno, models_obj
- [ ] models -> models_eval 