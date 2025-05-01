import os
from os import path as osp

import torch
import numpy as np

import engine
from .base import BaseSolver
from qt.data import ObjaverseDataModule


class EvaluationSolver(BaseSolver):

    def __init__(self, dm: ObjaverseDataModule, solver: BaseSolver):
        super().__init__()

        self.dm = dm
        self.solver = solver

        self.output_dir = engine.to_experiment_dir('outputs', self.dm.name)
        os.makedirs(self.output_dir, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self.solver(batch)
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
       
        self.plcc_metric(preds_normalized, labels_normalized)
        self.srocc_metric(preds_normalized, labels_normalized)
        self.krocc_metric(preds_normalized, labels_normalized)
        self.rmse_metric(preds_normalized, labels_normalized)

    def on_validation_epoch_end(self):
        
        plcc = self.plcc_metric.compute()
        srocc = self.srocc_metric.compute()
        krocc = self.krocc_metric.compute()
        rmse = self.rmse_metric.compute()

        result_str = f"[{self.dm.criterion}] PLCC: {plcc}, SROCC: {srocc}, KROCC: {krocc}, RMSE: {rmse}"
        print(result_str)
        out_path = osp.join(self.out_dir, f'eval_results_{self.dm.criterion}.txt')
        
        with open(out_path, 'w') as f:
            f.write(result_str)

