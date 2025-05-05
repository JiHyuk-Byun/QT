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
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 block_lr_scale: float = 0.1,
                 
                 ## loss
                 l1_w: float = 1,
                 rank_w: float = None,
                 rank_w_min: float = 1,
                 rank_w_max: float = 10,
                 warmup_start_step: int = 3000,
                 warmup_steps: int = 5000,
                 hard_thred: float = 1,
                 use_margin: bool = False,

                 save_ckpt_freq: int = 50,


                 ):
        super().__init__()

        self.dm = dm
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
#        self.warm_up_epochs = warm_up_epochs
        self.block_lr_scale = block_lr_scale
        self.scheduler_config = scheduler_config
        self.steps_per_epoch = len(self.dm.train_dataloader())
        self.loss_fn = L1RankLoss(l1_w=l1_w, rank_w=rank_w, rank_w_min=rank_w_min, rank_w_max=rank_w_max, 
                                  warmup_start_step=warmup_start_step, warmup_steps=warmup_steps,
                                  hard_thred=hard_thred, use_margin=use_margin)

        self._all_preds = []
        self._all_labels = []

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

        l1_loss, rank_loss, total_loss = self.loss_fn(preds, labels, self.global_step)

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

        self._all_preds.append(preds)
        self._all_labels.append(labels)
        # preds_normalized = self._min_max_normalize(preds)
        # labels_normalized = self._min_max_normalize(labels)
        # print(f'preds_norm: {preds_normalized[:20]}')
        # print(f'MOS_norm: {labels_normalized[:20]}')
        #loss = self.loss_fn(preds, labels)

    def on_validation_epoch_end(self):

        preds = np.concatenate(self._all_preds, axis=0)
        labels = np.concatenate(self._all_labels, axis=0)

        preds_norm = self._min_max_normalize(preds)
        labels_norm = self._min_max_normalize(labels)
        preds_norm_t = torch.from_numpy(preds_norm).to(self.device)
        labels_norm_t = torch.from_numpy(labels_norm).to(self.device)

        metrics_no_fitted = self._evaluate_metrics(preds_norm_t, labels_norm_t)
        
        _, _, preds_fitted = self._logistic_4_fitting(preds, labels)

        preds_t = torch.from_numpy(preds_fitted).to(self.device)
        labels_t = torch.from_numpy(labels).to(self.device)
        
        metrics_fitted = self._evaluate_metrics(preds_t, labels_t)


        self.log('val/plcc_no_fitted', metrics_no_fitted['plcc'], rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/srocc_no_fitted', metrics_no_fitted["srocc"], rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/krocc_no_fitted', metrics_no_fitted["krocc"], rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/rmse_no_fitted', metrics_no_fitted['rmse'], rank_zero_only=True, on_epoch=True, sync_dist=True)

        self.log('val/plcc_fitted', metrics_fitted['plcc'], rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/srocc_fitted', metrics_fitted["srocc"], rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/krocc_fitted', metrics_fitted["krocc"], rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/rmse_fitted', metrics_fitted['rmse'], rank_zero_only=True, on_epoch=True, sync_dist=True)

        self.track_score(metrics_no_fitted['srocc'])

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
            kwargs['T_max'] = self.total_epochs
            if warm_up > 0:
                num_training_steps = steps_per_epoch * self.total_epochs
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
    def __init__(self,
                 l1_w: float = 1,
                 rank_w: float = None,          
                 rank_w_min: float = 1,
                 rank_w_max: float = 10,
                 warmup_start_step: int = 3000,
                 warmup_steps: int = 5_000,
                 hard_thred: float = 1,
                 use_margin: bool = False,
                 **kwargs):                    

        super().__init__()
        # For checkpoint consistency
        if rank_w is not None:
            rank_w_min = rank_w
            rank_w_max = rank_w

        self.l1_w = l1_w
        self.rank_w_min = rank_w_min
        self.rank_w_max = rank_w_max
        self.warmup_start_step = warmup_start_step
        self.warmup_steps = warmup_steps
        self.hard_thred = hard_thred
        self.use_margin = use_margin
        
        self.register_buffer("rank_w", torch.tensor(self.rank_w_min))
        self.l1_loss = nn.SmoothL1Loss()
    
    def update_weight(self, global_step):
        s = max(0.0, min(1.0, (global_step - self.warmup_start_step)/ self.warmup_steps))
        
        new_val = self.rank_w_min + (self.rank_w_max - self.rank_w_min) * s                     # float or tensor OK
        
        self.rank_w.data.fill_(new_val)
    
    
    def forward(self, preds, gts, global_step = None):

        if global_step is not None:
            self.update_weight(global_step)

        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = self.l1_loss(preds, gts) * self.l1_w

        # simple rank
        if float(self.rank_w) > 0:
            n = len(preds)
            diff_label = gts[:, None] - gts[None, :]             # (n,n)
            diff_pred  = preds[:, None] - preds[None, :]

            sign = torch.sign(diff_label)
            masks_hard = (torch.abs(diff_label) < self.hard_thred) & (torch.abs(diff_label) > 0)
            
            if self.use_margin:
                core = torch.relu(torch.abs(diff_label) - sign * (diff_pred))
            else:
                core = torch.relu(- sign * (diff_pred))
            
            denom = masks_hard.sum().clamp(min=1)
            rank_loss = (core * masks_hard).sum() / denom
        
        else:
            rank_loss = preds.new_tensor(0.0)

        
        loss_total = l1_loss + rank_loss * float(self.rank_w)
        
        return l1_loss, rank_loss, loss_total
