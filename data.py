import json
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import torch
import torchvision
import numpy as np


class SyntheticDataset(Dataset):
    def __init__(self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        split: str = "train", #mode
        n_steps=10, 
        dataset_class='spmot', 
        T=0): #transforms, path

        assert dataset_class in ['vmds', 'vor', 'spmot']
        self.clevr_transforms = clevr_transforms
        #self.split_dict = {"train":"train2017", "val":"val2017", "test":"test2017"}
        self.data_root = data_root
        self.split = split
        

        
        imgs = np.load(os.path.join(data_root, dataset_class, '{}_{}.npy'.format(dataset_class, split)))
        imgs = imgs[:, :n_steps]
        imgs = imgs[0:max_num_images,:,:,:,:]
        if T and T < n_steps:
            imgs = np.concatenate(np.split(imgs, imgs.shape[1]//T, axis=1))
        self.imgs = imgs
        #self.imgs = [img for img in imgs]
        self.num_samples = self.imgs.shape[0]


    def __getitem__(self, index):
        x = self.imgs[index]
        fr, ch, imx, imy = x.shape
        x = torch.stack([transforms.ToTensor()(x[i]) for i in range(fr)])
        x = x.permute(0,2,1,3)
        return x.float()

    def __len__(self):
        return self.num_samples


class CLEVRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clevr_transforms: Callable,
        num_workers: int,
        n_steps: int,
        dataset_class: str,
        T: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images
        self.n_steps= n_steps
        self.dataset_class = dataset_class
        self.T = T

        self.train_dataset = SyntheticDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clevr_transforms=self.clevr_transforms,
            split="train",
            n_steps = self.n_steps,
            dataset_class=self.dataset_class, 
            T=self.T
          )
        self.val_dataset = SyntheticDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clevr_transforms=self.clevr_transforms,
            split="val",
            n_steps = self.n_steps,
            dataset_class=self.dataset_class, 
            T=self.T
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

# This is not being used, actual ones being used are in train.py
class CLEVRTransforms(object):
    def _init_(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def _call_(self, input, *args, **kwargs):
        return self.transforms(input)