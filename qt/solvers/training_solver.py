from os import path as osp

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef, KendallRankCorrCoef

from qt.data import ObjaverseDataModule
from .base import BaseSolver
from qt.metrics import PLCC, SROCC, KROCC, RMSE

class Ptv3Solver(BaseSolver):

    def __init__(self, dm: ObjaverseDataModule, model: nn.Module,
                 
                 ## optimizer, scheduler params
                 scheduler_config: dict,
                 epochs: int = 300,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 block_lr_scale: float = 0.1,

                 save_ckpt_freq: int = 50,
                 ):
        super().__init__()

        self.dm = dm
        self.model = model

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
#        self.warm_up_epochs = warm_up_epochs
        self.block_lr_scale = block_lr_scale
        self.scheduler_config = scheduler_config
        self.steps_per_epoch = len(self.dm.train_dataloader())
        self.loss_fn = nn.SmoothL1Loss()
        self.plcc_metric = PearsonCorrCoef() #PLCC()
        self.srocc_metric = SpearmanCorrCoef() #SROCC()
        self.krocc_metric = KendallRankCorrCoef(variant='b')#KROCC()
        self.rmse_metric = RMSE()

        self.save_ckpt_freq = save_ckpt_freq
        
    def configure_optimizers(self):
        optimizer = [AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)]
        
        scheduler = [self._get_scheduler(optimizer[0], self.scheduler_config, steps_per_epoch=self.steps_per_epoch)]

        return optimizer, scheduler
    
    def forward(self, batch):

        outputs = self.model(batch)

        return outputs
    
    def training_step(self, batch, batch_idx):
        #print(batch['feat'].shape)
        try:
            outputs = self(batch)
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print(f"Out of MemoryError with input shape: {batch['feat'].shape}")
            exit()
        preds = outputs
        labels = batch['mos']

        loss = self.loss_fn(preds, labels)
        self.log('train/loss', loss.mean().item())#, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            'loss': loss,
        }
    
    def validation_step(self, batch, batch_idx):
        try:
            outputs = self(batch)
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print(f"Out of MemoryError with input shape: {batch['feat'].shape}")
        preds = outputs
        labels = batch['mos']
        print(f'preds: {preds[:20]}')
        print(f'MOS: {labels[:20]}')
        preds_normalized = self._min_max_normalize(preds)
        labels_normalized = self._min_max_normalize(labels)
        print(f'preds_norm: {preds_normalized[:20]}')
        print(f'MOS_norm: {labels_normalized[:20]}')
        #loss = self.loss_fn(preds, labels)

        self.plcc_metric(preds_normalized, labels_normalized)
        self.srocc_metric(preds_normalized, labels_normalized)
        self.krocc_metric(preds_normalized, labels_normalized)
        self.rmse_metric(preds_normalized, labels_normalized)

    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        plcc = self.plcc_metric.compute()
        srocc = self.srocc_metric.compute()
        krocc = self.krocc_metric.compute()
        rmse = self.rmse_metric.compute()

        self.log('val/plcc', plcc, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/srocc', srocc, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/krocc', krocc, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/rmse', rmse, rank_zero_only=True, on_epoch=True, sync_dist=True)

        self.track_score(srocc)

        # if (epoch + 1) % self.save_ckpt_freq == 0:
        #     ckpt_path = osp.join(self.out_dir, f'epoch-{epoch+1:03d}.ckpt')
        #     self.trainer.save_checkpoint(ckpt_path)
        #     print(f'Save checkpoint: {ckpt_path}')
    
    @torch.no_grad()
    def _min_max_normalize(self, mos_array):
        return (mos_array - mos_array.min()) / (mos_array.max() - mos_array.min()) * 100

    def _get_scheduler(self, optimizer, scheduler_config, steps_per_epoch=None):
        scheduler_type = scheduler_config["type"]
        kwargs = {k: v for k, v in scheduler_config.items() if k != "type"}

        if scheduler_type == "OneCycleLR":
            if steps_per_epoch is not None:
                kwargs["total_steps"] = steps_per_epoch
            return OneCycleLR(optimizer=optimizer, **kwargs)

        elif scheduler_type == "CosineAnnealingLR":
            warm_up = kwargs.pop("warm_up", 0) 
            kwargs['T_max'] = self.epochs
            if warm_up > 0:
                num_training_steps = steps_per_epoch * self.epochs
                num_warmup_steps = int(num_training_steps * warm_up)
               
                return get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            else:    
                return CosineAnnealingLR(optimizer=optimizer, **kwargs)
        
        elif scheduler_type == "StepLR":
            return StepLR(optimizer=optimizer, **kwargs)

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")