# Data2


## Post processing

1. Download GT from NAS, and extract it.

```
```

2. generate ground truths

```
conda activate bop_toolkit && cd ~/Workspace/papers/2022/clora/bop_toolkit
conda activate bop_toolkit && cd ~/Workspace/clora/bop_toolkit


# real
python aihub/data2/6_calc_gt_masks_real.py  --n_scenes 200 --n_proc 5 --proc 1
python aihub/data2/8_calc_gt_order.py --is_real --n_scenes 200 --n_proc 5 --proc 1


# synthetic
python aihub/data2/8_calc_gt_orders.py --n_scenes 10 --n_proc 10 --proc 6

```