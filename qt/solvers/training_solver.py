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
        self.criterion = self.dm.criterion
        self.lr = lr
        self.weight_decay = weight_decay
#        self.warm_up_epochs = warm_up_epochs
        self.block_lr_scale = block_lr_scale
        self.scheduler_config = scheduler_config
        self.steps_per_epoch = len(self.dm.train_dataloader())
        self.loss_fn = L1RankLoss(l1_w=l1_w, 
                                  rank_w=rank_w, 
                                  rank_w_min=rank_w_min, 
                                  rank_w_max=rank_w_max, 
                                  warmup_start_step=warmup_start_step, 
                                  warmup_steps=warmup_steps,
                                  hard_thred=hard_thred, 
                                  use_margin=use_margin,
                                  num_criterion=len(self.criterion))

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
        outputs = self(batch)
        preds = outputs
        B, C = preds.shape
        labels = batch['mos']
        labels = labels.view(B, C)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        l1_loss, rank_loss, total_loss = self.loss_fn(preds, labels, self.global_step)

        self.log('train/l1_loss', l1_loss.mean().item())
        self.log('train/rank_loss', rank_loss.mean().item())
        self.log('train/loss', total_loss.mean().item())#, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            'loss': total_loss,
        }
    
    def validation_step(self, batch, batch_idx):

        outputs = self(batch)
        preds = outputs
        B, C = preds.shape
        labels = batch['mos']
        labels = labels.view(B, C)

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

        preds_local = torch.cat(self._all_preds, dim=0)
        labels_local = torch.cat(self._all_labels, dim=0)
        # print('local', preds_local)
        # preds_all = self.all_gather(preds_local)
        # labels_all = self.all_gather(labels_local)

        # if self.global_rank != 0:
        #     return
        # print(preds_all)
        # assert len(self.criterion) == preds_all.shape[-1]
        # preds_all = preds_all.reshape(-1, len(self.criterion))
        # labels_all = labels_all.reshape(-1, len(self.criterion))
        sroccs = []

        for i, crit in enumerate(self.criterion):
            pred = preds_local[:, i]
            gt = labels_local[:, i]
            print('pred: ', pred)
            print('labels:', gt)

            pred_norm = self._min_max_normalize(pred)
            gt_norm = self._min_max_normalize(gt)

            metrics_no_fitted = self._evaluate_metrics(pred_norm, gt_norm)
            for k, v in metrics_no_fitted.items():
                self.log(f'val/{crit}/{k}_no_fitted', v, rank_zero_only=True, on_epoch=True, sync_dist=True)
            
            sroccs.append(metrics_no_fitted['srocc'])

            # _, _, pred_fitted = self._logistic_4_fitting(pred.detach().cpu().numpy(), gt.detach().cpu().numpy())
            
            # preds_t = torch.from_numpy(pred_fitted).to(self.device)
            # gt_t = gt.float()
            # metrics_fitted = self._evaluate_metrics(preds_t, gt_t)
            # for k, v in metrics_fitted.items():
            #     self.log(f'val/{crit}/{k}_fitted', v, rank_zero_only=True, on_epoch=True, sync_dist=True)
        
        mean_srocc = torch.stack(sroccs).mean()
        self.track_score(mean_srocc)

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
                 num_criteria: int = 1,
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
        self.num_criteria = num_criteria

        self.register_buffer("rank_w", torch.tensor(self.rank_w_min))
        self.l1_loss = nn.SmoothL1Loss(reduction='none')
    
    def update_weight(self, global_step):
        s = max(0.0, min(1.0, (global_step - self.warmup_start_step)/ self.warmup_steps))
        
        new_val = self.rank_w_min + (self.rank_w_max - self.rank_w_min) * s                     # float or tensor OK
        
        self.rank_w.data.fill_(new_val)
    
    
    def forward(self, preds, gts, global_step = None):

        if global_step is not None:
            self.update_weight(global_step)

        B, C = preds.shape
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_elem = self.l1_loss(preds, gts)
        l1_loss = l1_elem.mean(dim=0) * self.l1_w

        # simple rank
        rank_losses = []
        if float(self.rank_w) > 0:
            for c in range(C):
                pi = preds[:, c]
                gi = gts[:, c]

                diff_label = gi[:, None] - gi[None, :]             # (n,n)
                diff_pred  = pi[:, None] - pi[None, :]

                sign = torch.sign(diff_label)
                masks_hard = (torch.abs(diff_label) < self.hard_thred) & (torch.abs(diff_label) > 0)
                
                if self.use_margin:
                    core = torch.relu(torch.abs(diff_label) - sign * (diff_pred))
                else:
                    core = torch.relu(- sign * (diff_pred))
                
                denom = masks_hard.sum().clamp(min=1)
                rank_losses.append((core * masks_hard).sum() / denom)
        
        else:
            rank_losses = [preds.new_tensor(0.0) for _ in range(C)]

        rank_loss = torch.stack(rank_losses)
        
        loss_per_criterion = l1_loss + rank_loss * float(self.rank_w) # [C]
        total_loss = loss_per_criterion.mean()

        return l1_loss.mean(), rank_loss.mean(), total_loss


