import os
import cv2
import numpy as np
import torch
import random
import json

from scipy.stats import multivariate_normal
from scipy.ndimage import center_of_mass

import albumentations as A
import torch.utils.data as data
import glob
from torchvision.transforms import Normalize
import imageio
import random


class CustomDataset(data.Dataset):

    def __init__(self, cfg, train=True):

        self.dataset_path = cfg["dataset_path"]
        self.img_size = cfg["img_size"]
        
        # transformation
        train_transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomBrightnessContrast(p=0.5),
            A.augmentations.transforms.ColorJitter(),
            A.augmentations.transforms.ChannelShuffle(),
            A.augmentations.transforms.RandomGamma(),
            A.augmentations.transforms.ImageCompression(),

        ],
        additional_targets={'ann': 'mask'}
        )
        test_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(self.img_size[1], self.img_size[0])
        ],
        additional_targets={'ann': 'mask'}
        )
        if train:
            self.transform = train_transform
            scene_ids = list(range(0, 90))
        else: 
            self.transform = test_transform
            scene_ids = list(range(90, 100))
        self.color_normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.mask_paths = []
        for scene_id in scene_ids:
            mask_paths = sorted(glob.glob(os.path.join(self.dataset_path, "{0:06d}".format(scene_id), "mask_visib", "*.png")))
            self.mask_paths += mask_paths


    def __getitem__(self, idx):

        try:
            mask_path = self.mask_paths[idx]
            scene_id = int(os.path.basename(os.path.dirname(os.path.dirname(mask_path))))
            im_id = os.path.basename(mask_path).split("_")[0]
            img_path = os.path.join(self.dataset_path, "{0:06d}".format(int(scene_id)), "rgb", "{0:06d}.jpg".format(int(im_id)))

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path) # 0: background, 255: foreground
            # get bounding box
            ys, xs, _ = np.where(mask)
            y1, y2 = np.min(ys), np.max(ys)
            x1, x2 = np.min(xs), np.max(xs)
            w = x2 - x1
            h = y2 - y1




            w_offset = int(w * random.random()*0.2)
            h_offset = int(h * random.random()*0.2)
            roi_xyxy = [x1-w_offset, y1-h_offset, x2+w_offset, y2+h_offset]
            roi_xyxy = [max(0, roi_xyxy[0]), max(0, roi_xyxy[1]), min(img.shape[1], roi_xyxy[2]), min(img.shape[0], roi_xyxy[3])]

            img_crop = img[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2], :]
            mask_crop = mask[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2], :]

            img_crop = cv2.resize(img_crop, (self.img_size[1], self.img_size[0]))
            mask_crop = cv2.resize(mask_crop, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

            transformed = self.transform(image=img_crop, ann=mask_crop)
            input = transformed["image"]
            gt = transformed["ann"]

            input = torch.tensor(np.transpose(input, (2, 0, 1)), dtype=torch.float32)
            input = self.color_normalize(input)

            # change gt to 2 channel (background, foreground)
            gt = torch.tensor(gt[:, :, 0]/255, dtype=torch.float32)
            gt = torch.stack([1-gt, gt], dim=-1)
            gt = torch.tensor(np.transpose(gt, (2, 0, 1)), dtype=torch.float32)

            return input, gt 
        except:
            idx = np.random.randint(0, len(self)-1)
            sample = self[idx]
            return sample
        
    def __len__(self):
        return len(self.mask_paths)

