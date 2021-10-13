import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from model import SlotAttentionModel
from params import SlotAttentionParams
from utils import Tensor
from utils import to_rgb_from_tensor

import matplotlib.pyplot as plt


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, params: SlotAttentionParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss
        
    def si(self):
        dl = self.datamodule.train_dataloader()
        #perm = torch.randperm(self.params.batch_size)
        #idx = perm[: self.params.n_samples]
        for (idx, batch) in enumerate(dl):
            print(idx)
            print(batch.shape)

    def magic_combine(self,x, dim_begin, dim_end):
        combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
        return x.view(combined_shape)

    def sample_images(self):
        #dl = self.datamodule.val_dataloader()
        dl = self.datamodule.train_dataloader()
        perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        batch = next(iter(dl))[idx]
        if self.params.gpus > 0:
            batch = batch.to(self.device)
        recon_combined, recons, masks, slots = self.model.forward(batch)
        
         

        #print(batch.unsqueeze(1).shape)
        #print(recon_combined.unsqueeze(1).shape)
        #print((recons * masks + (1 - masks)).shape)
        #
        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, num_slots, frames, C, H, W = recons.shape
        print(out.shape)
        out = out[0,:,:,:,:,:]
        # recons_new = recons[:,:,-1,:,:,:]
        #print("Here")
        # print(recons_new.shape)
        # out = out.permute(1,3,4,5,2,0)
        # out = out.reshape(num_slots+2, C, H, W, -1)
        # out = out.permute(4,0,1,2,3)
        images = vutils.make_grid(
            out.reshape(frames * out.shape[0], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
                #factor *= self.params.scheduler_gamma ** (step / decay_steps)###
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
