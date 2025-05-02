import os
from os import path as osp
from typing import List

import torch
from lightning.pytorch import LightningModule, Callback
from torchmetrics import Metric
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef, KendallRankCorrCoef

import engine
from qt.metrics import PLCC, SROCC, KROCC, RMSE

class BaseSolver(LightningModule):

    def __init__(self):
        super().__init__()

        #self.total_epochs = None
        self._best_score = -float('inf')
        self._additional_callbacks: List[Callback] = [_DefaultTaskCallback()]
        self._metrics = None
        self.checkpoint_epoch = -1
        self.out_dir = engine.to_experiment_dir('outputs')
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.plcc_metric = PearsonCorrCoef() #PLCC()
        self.srocc_metric = SpearmanCorrCoef() #SROCC()
        self.krocc_metric = KendallRankCorrCoef(variant='b')#KROCC()
        self.rmse_metric = RMSE()

    def configure_callbacks(self):
        return self._additional_callbacks
    
    def add_callback(self, callback: Callback):
        self._additional_callbacks.append(callback)
    
    @property
    def best_score(self):
        return self._best_score
    
    @property
    def total_epochs(self):
        return self.trainer.max_epochs if self.trainer else None
    
    # def on_fit_start(self):
    #     self.total_epochs = self.trainer.max_epochs
    
    def track_score(self, score: any):
        if score > self._best_score:
            self._best_score = score
            # epoch = self.current_epoch
            # best_path = osp.join(self.out_dir, f"best-{epoch}.ckpt")
            # self.trainer.save_checkpoint(best_path)
            # print(f"[BEST] epoch {epoch} val_srocc={score} saved to {best_path}")

        self.log('score', score, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('best', self._best_score, rank_zero_only=True, on_epoch=True, sync_dist=True)

    def load_checkpoint(self, checkpoint_path: str):
        print(f'checkpoint loaded: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict, strict=False)
        self.checkpoint_epoch = checkpoint['epoch']

    def reset_metrics(self):
        if self._metrics is None:
            self._metrics = [value for value in self.modules() if isinstance(value, Metric)]

        for metric in self._metrics:
            metric.reset()

    def reset_parameters(self):
        pass
    
    @torch.no_grad()
    def _min_max_normalize(self, mos_array):
        return (mos_array - mos_array.min()) / (mos_array.max() - mos_array.min()) * 100
    
class _DefaultTaskCallback(Callback):
    def on_sanity_check_end(self, trainer, solver):
        if not isinstance(solver, BaseSolver):
            return
        solver._best_score = -float('inf')

    def on_validation_epoch_start(self, trainer, solver) -> None:
        if not isinstance(solver, BaseSolver):
            return
        solver.reset_metrics()

        

        