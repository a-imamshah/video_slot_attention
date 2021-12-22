import json
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils import compact


class CLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_images = max_num_images
        self.data_path = os.path.join(data_root, "vor", split)
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        files = self.get_files()
        nFrames = 10
        self.files = [files[nFrames*i:nFrames*(i+1)] for i in range(len(files)//nFrames)]

    def __getitem__(self, index: int):
        image_paths = self.files[index]
        imgs = [Image.open(image_path) for image_path in image_paths]
        imgs = [img.convert("RGB") for img in imgs]
        imgs = torch.stack([self.clevr_transforms(img) for img in imgs])
        return imgs

    def __len__(self):
        return len(self.files)

    def get_files(self) -> List[str]:
        paths = [f for f in os.listdir(self.data_path) if f.endswith(".jpg")]
        paths = [os.path.join(self.data_root, "vor", self.split, f) for f in paths]
        return sorted(compact(paths))


class CLEVRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clevr_transforms: Callable,
        max_n_objects: int,
        num_workers: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images

        self.train_dataset = CLEVRDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clevr_transforms=self.clevr_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
        )
        self.val_dataset = CLEVRDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clevr_transforms=self.clevr_transforms,
            split="val",
            max_n_objects=self.max_n_objects,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CLEVRTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
