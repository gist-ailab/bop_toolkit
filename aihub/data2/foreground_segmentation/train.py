import os
import sys
import cv2
from tqdm import tqdm
import yaml
import argparse
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from aihub.data2.foreground_segmentation.loader import CustomDataset
from aihub.data2.foreground_segmentation.loss import *
from aihub.data2.foreground_segmentation.util import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="base", help='config file name')
    parser.add_argument('--gpu_id', default="0", help='gpu_id')
    args = parser.parse_args()

    with open('aihub/data2/foreground_segmentation/configs/{}.yaml'.format(args.config)) as f:
        cfg = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    '''load dataset'''
    train_dataset = CustomDataset(cfg)
    test_dataset = CustomDataset(cfg, train=False)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                                    shuffle=True, num_workers=1)
    val_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                    num_workers=2)

    '''load model'''
    model = smp.create_model(
            arch = cfg["arch"],
            encoder_name = cfg["encoder_name"],
            in_channels = 3,
            classes = 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using {} gpu!".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Loss = getattr(sys.modules[__name__], cfg["loss"])(**cfg["loss_kwargs"])
    best_itr, best_score = 0, 0
    best_metrics = get_initial_metric(best=True)
    total_itr = 0
    for epoch in range(cfg["maximum_epoch"]):
        model = model.train()
        pbar = tqdm(train_dataloader)
        train_metrics = get_initial_metric()
        for itr, (input, gt) in enumerate(pbar):

            optimizer.zero_grad()
            pred = model(input.cuda())
            gt = gt.cuda()
            loss = Loss(pred, gt)
            loss.backward()
            optimizer.step()

            # calculate metric
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
            train_metrics = compute_metrics(pred, gt, train_metrics)
            pbar.set_postfix({'train_loss': loss.item(), 'train_iou': train_metrics['iou'][-1]})
            total_itr += 1

            if total_itr % cfg["test_interval"] == 0:
                train_metrics = get_average_metrics(train_metrics)    
                for key in train_metrics.keys():
                    print("train | {:<15}| {:<7.3f}".format(key, train_metrics[key]))

                model = model.eval()
                val_metrics = get_initial_metric()
                
                for itr, (input, gt) in enumerate(tqdm(val_dataloader)):
                    with torch.no_grad():
                        pred = model(input.cuda())
                        pred = torch.sigmoid(pred)
                        pred = pred > 0.5
                        gt = gt.cuda()
                    val_metrics = compute_metrics(pred, gt, val_metrics)
                    if itr > 100:
                        break
                val_metrics = get_average_metrics(val_metrics)    

                if best_score < val_metrics["iou"]:
                    best_score, best_metrics, best_itr = val_metrics["iou"], val_metrics, total_itr
                    torch.save(model.state_dict(), os.path.join(cfg["log_dir"], f"itr_{total_itr}.pkl"))
                        
                print("epoch: {}| loss: {:.4f}, best_itr: {}, best_score: {:.4f}".format(epoch, loss.item(), best_itr, best_score))
                for key in val_metrics.keys():
                    print("val | {:<15}| {:<7.3f}| best {:<15}| {:<7.3f}".format(key, val_metrics[key], key, best_metrics[key]))
                model = model.train()
                train_metrics = get_initial_metric()
