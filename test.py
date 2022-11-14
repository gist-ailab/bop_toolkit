import json

for i in range(10):

    path = "/home/ailab/OccludedObjectDataset/ours/data2/data2_syn_source/train_pbr/{:06d}/occ_mat.json".format(int(i))
    with open(path, 'r') as f:
        file = json.load(f)

    not_processed = []
    for j in range(1000):
        if str(j) not in file.keys():
            not_processed.append(i)

    print("{}: ".format(i), len(not_processed))

    python aihub/data2/12_calc_gt_order_.py --n_proc 8 --proc 2
