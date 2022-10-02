# Data2


## Post processing

1. Download GT from NAS, and extract it.

```
```

2. generate ground truths

```
conda activate bop_toolkit && cd ~/Workspace/papers/2022/clora/bop_toolkit

# real
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 1
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 2
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 3
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 4
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 5
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 6
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 7
python aihub/data2/6_calc_gt_masks_and_orders.py --is_real --n_scenes 200 --n_proc 8 --proc 8


# synthetic
python aihub/data2/6_calc_gt_masks_and_orders.py --n_scenes 10 --n_proc 10 --proc 1
```