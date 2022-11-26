import torch
import copy
import numpy as np


def compute_metrics(preds, gts, metrics):
    
    preds = preds.detach()
    gts = gts.detach()

    intersection = torch.logical_and(preds, gts)
    union = torch.logical_or(preds, gts)
    precision = torch.sum(intersection) / torch.sum(preds)
    recall = torch.sum(intersection) / torch.sum(gts)
    iou = torch.sum(intersection) / torch.sum(union)

    metrics["prec"].append(precision.item())
    metrics["recall"].append(recall.item())
    metrics["iou"].append(iou.item())
    return metrics

def get_initial_metric(best=False):
    default = []
    if best:
        default = 0
    metrics = {"iou": copy.deepcopy(default), "prec": copy.deepcopy(default), "recall": copy.deepcopy(default)}
    return metrics    

def get_average_metrics(metrics):
    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])
    return metrics