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
        
        self._all_preds = []
        self._all_labels = []

    def on_validation_epoch_start(self) -> None:
        self._all_preds.clear()
        self._all_labels.clear()
        self.reset_metrics()

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self.solver(batch)
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print(f"Out of MemoryError with input shape: {batch['feat'].shape}")
        preds = outputs.detach().cpu().numpy()
        labels = batch['mos'].detach().cpu().numpy()
        print(f'preds: {preds[:20]}')
        print(f'MOS: {labels[:20]}')
        self._all_preds.append(preds)
        self._all_labels.append(labels)
#        _, _, preds_normalized = self._logistic_4_fitting(preds)
#        labels_normalized = self._logistic_4_fitting(labels)
        # print(f'preds_norm: {preds_normalized[:20]}')
        # print(f'MOS_norm: {labels[:20]}')
       


    def on_validation_epoch_end(self):
        
        preds = np.concatenate(self._all_preds, axis=0)
        labels = np.concatenate(self._all_labels, axis=0)


        preds_norm = self._min_max_normalize(preds)
        labels_norm = self._min_max_normalize(labels)

        preds_norm_t = torch.from_numpy(preds_norm).to(self.device)
        labels_norm_t = torch.from_numpy(labels_norm).to(self.device)
        
        metrics_no_fitted = self._evaluate_metrics(preds_norm_t, labels_norm_t)


        _, _, preds_fitted = self._logistic_4_fitting(preds, labels)
        print('preds:', preds)
        print('preds_fitted', preds_fitted)
        print('labels: ', labels)
        preds_t = torch.from_numpy(preds_fitted).to(self.device)
        labels_t = torch.from_numpy(labels).to(self.device)
        
        metrics_fitted = self._evaluate_metrics(preds_t, labels_t)
        
        result_str_no_fitted = f"[{self.dm.criterion}] PLCC: {metrics_no_fitted['plcc']}, SROCC: {metrics_no_fitted['srocc']}, KROCC: {metrics_no_fitted['krocc']}, RMSE: {metrics_no_fitted['rmse']}"
        result_str_fitted = f"[{self.dm.criterion}] PLCC: {metrics_fitted['plcc']}, SROCC: {metrics_fitted['srocc']}, KROCC: {metrics_fitted['krocc']}, RMSE: {metrics_fitted['rmse']}"
        
        result = f'---No fitted Result---\n {result_str_no_fitted}' + '\n' + f'---fitted Result---\n {result_str_fitted}'
        print(result)
        out_path = osp.join(self.out_dir, f'eval_results_{self.dm.criterion}.txt')
        
        with open(out_path, 'w') as f:
            f.write(result)

    def _evaluate_metrics(self, preds, labels):
        self.reset_metrics()
        self.plcc_metric(preds, labels)
        self.srocc_metric(preds, labels)
        self.krocc_metric(preds, labels)
        self.rmse_metric(preds, labels)
        
        plcc = self.plcc_metric.compute()
        srocc = self.srocc_metric.compute()
        krocc = self.krocc_metric.compute()
        rmse = self.rmse_metric.compute()

        return {'plcc': plcc,
                'srocc': srocc,
                'krocc': krocc,
                'rmse': rmse}

