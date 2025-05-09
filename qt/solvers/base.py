import os
from os import path as osp
from typing import List

import torch
from lightning.pytorch import LightningModule, Callback
from torchmetrics import Metric
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef, KendallRankCorrCoef, MeanSquaredError
import numpy as np
from scipy.optimize import curve_fit


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
        
        self.plcc_metric = PearsonCorrCoef(sync_on_compute=True) #PLCC()
        self.srocc_metric = SpearmanCorrCoef(sync_on_compute=True) #SROCC()
        self.krocc_metric = KendallRankCorrCoef(variant='b',sync_on_compute=True)#KROCC()
        self.rmse_metric = MeanSquaredError(squared=False, sync_on_compute=True)
        
        self._all_preds = []
        self._all_labels = []
        
    def on_validation_epoch_start(self) -> None:
        self._all_preds.clear()
        self._all_labels.clear()
        self.reset_metrics()

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
        self.load_state_dict(state_dict, strict=True)

        #self.checkpoint_epoch = checkpoint['epoch']

    def reset_metrics(self):
        if self._metrics is None:
            self._metrics = [value for value in self.modules() if isinstance(value, Metric)]

        for metric in self._metrics:
            metric.reset()

    def reset_parameters(self):
        pass
    
    @torch.no_grad()
    def _min_max_normalize(self, mos_array: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return (mos_array - mos_array.min()) / (mos_array.max() - mos_array.min() + eps) * 100
    
    @torch.no_grad()
    def _logistic_4_fitting(self, x, y):
        def func(x, b0, b1, b2, b3):
            return b1 + np.divide(b0 - b1, 1 + np.exp(np.divide(b2 - x, np.abs(b3))))
        x_axis = np.linspace(np.amin(x), np.amax(x), 100)
        init = np.array([np.max(y), np.min(y), np.mean(x), 0.1])
        popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
        curve = func(x_axis, *popt)
        fitted = func(x, *popt)

        return x_axis, curve, fitted

    def _evaluate_metrics(self, preds, labels):
        self.reset_metrics()

        self.plcc_metric.to(preds.device)
        self.srocc_metric.to(preds.device)
        self.krocc_metric.to(preds.device)
        self.rmse_metric.to(preds.device)

        self.plcc_metric.update(preds, labels)
        self.srocc_metric.update(preds, labels)
        self.krocc_metric.update(preds, labels)
        self.rmse_metric.update(preds, labels)
        
        plcc = self.plcc_metric.compute()
        srocc = self.srocc_metric.compute()
        krocc = self.krocc_metric.compute()
        rmse = self.rmse_metric.compute()

        return {'plcc': plcc,
                'srocc': srocc,
                'krocc': krocc,
                'rmse': rmse}

class _DefaultTaskCallback(Callback):
    def on_sanity_check_end(self, trainer, solver):
        if not isinstance(solver, BaseSolver):
            return
        solver._best_score = -float('inf')

    def on_validation_epoch_start(self, trainer, solver) -> None:
        if not isinstance(solver, BaseSolver):
            return
        solver.reset_metrics()

        
class MinMaxWrapper(Metric):
    def __init__(self, base_metric_cls, **kwargs):
        super().__init__(compute_on_step=False)
        self.base = base_metric_cls(**kwargs, compute_on_step=False)
        self.add_state("preds",  default=[], dist_reduce_fx="cat")
        self.add_state("targets",default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self):
        p = torch.cat(self.preds);  t = torch.cat(self.targets)
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return self.base(p, t)
        