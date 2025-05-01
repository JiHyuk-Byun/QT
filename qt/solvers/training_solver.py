from os import path as osp

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from qt.data import ObjaverseDataModule
from .base import BaseSolver


class Ptv3Solver(BaseSolver):

    def __init__(self, dm: ObjaverseDataModule, model: nn.Module,
                 
                 ## optimizer, scheduler params
                 scheduler_config: dict,
                 epochs: int = 300,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 block_lr_scale: float = 0.1,
                 
                 ## loss
                 l1_w: float = 1,
                 rank_w: float = 1,
                 hard_thred: float = 1,
                 use_margin: bool = False,

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
        self.loss_fn = L1RankLoss(l1_w=l1_w, rank_w=rank_w, hard_thred=hard_thred, use_margin=use_margin)


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

        l1_loss, rank_loss, total_loss = self.loss_fn(preds, labels)

        self.log('train/l1_loss', l1_loss.mean().item())
        self.log('train/rank_loss', rank_loss.mean().item())
        self.log('train/loss', total_loss.mean().item())#, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            'loss': total_loss,
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
        
class L1RankLoss(torch.nn.Module):
    """
    L1 loss + Rank loss
    """

    def __init__(self, 
                 l1_w: float = 1,
                 rank_w: float = 1,
                 hard_thred = 1,
                 use_margin = False,):
        super(L1RankLoss, self).__init__()
        self.l1_w = l1_w
        self.rank_w = rank_w
        self.hard_thred = hard_thred
        self.use_margin = use_margin

        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = self.l1_loss(preds, gts) * self.l1_w

        # simple rank
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w

        
        return l1_loss, rank_loss, loss_total