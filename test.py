from typing import Optional

import json
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from data import CLEVRDataModule
from method import SlotAttentionMethod
from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import ImageLogCallback
from utils import rescale

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import torch
import torchvision
import numpy as np

import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import Tensor
from utils import to_rgb_from_tensor

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
        #print(x.shape)
        #print(index)
        x = x / 255.0
        fr, ch, imx, imy = x.shape
        x = x.reshape((fr*ch, imx, imy))
        #print(x.shape)
        if self.clevr_transforms is not None:
            x = self.clevr_transforms(x)
        x = x.reshape((fr,ch, imx, imy))
        #print(x.shape)
        return x

    def __len__(self):
        return self.num_samples

def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({params.num_slots - 1}) objects.")
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Lambda(rescale),  # rescale between -1 and 1
            #transforms.Resize(params.resolution),
        ]
    )
    
    train_dataset = SyntheticDataset(
            data_root=params.data_root,
            max_num_images=params.num_train_images,
            clevr_transforms=clevr_transforms,
            split="train",
            n_steps = params.n_steps,
            dataset_class=params.dataset_class, 
            T=params.T
          )

    dl= DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers)
    
    
    # for (idx, batch) in enumerate(dl):
    #     print(idx)
    #     print(batch.shape)
        
    # print(len(dl))

    perm = torch.randperm(params.batch_size)
    idx = perm[: params.n_samples]
    batch = next(iter(dl))[idx]
    if params.gpus > 0:
        batch = batch.cuda()
    
    out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                ],
                dim=1,
            )
        )

    batch_size, num_slots, frames, C, H, W = out.shape
    print(out.shape)
    out = out[:,:,-1,:,:,:]

    images = vutils.make_grid(out.reshape(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],)

    
    
    
    
    
if __name__ == "__main__":
    main()
