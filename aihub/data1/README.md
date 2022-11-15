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

models: ply (colored points + texture) compatible with bop's models 
- w/ texture -> x, y, z, nx, ny, nz, texture_u, texture_v, face, vertex_indices
- w/o texture -> x, y, z, nx, ny, nz, r, g, b, a, vertex_indices
models_colored: ply (colored points)
models_anno: downsampled ply (colored points) for 6d object pose annotation
models_eval: ply (colored points + texture) compatible with bop's models_eval
models_obj: obj (colored points + texture) for aihub submission 
```


- [x] HOPE -> models, models_eval
- [X] YCBV -> models, models_eval
- [X] YCB  -> models, models_eval
- [X] GraspNet -> models
- [X] APC -> models
- [X] Artec -> models
- [X] Modify H of object 10, 11, 13, 18, 66, 69, 72, 75, 82
- [ ] models -> models_anno
- [X] models -> models_obj
- [X] models -> models_eval
- [X] calculate models_info.json
- [ ] define obj_symmetries