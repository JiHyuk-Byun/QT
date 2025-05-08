import os
from os import path as osp

import torch
import numpy as np

import engine
from .base import BaseSolver
from qt.data import QA3DBaseDataModule


class EvaluationSolver(BaseSolver):

    def __init__(self, dm: QA3DBaseDataModule, solver: BaseSolver):
        super().__init__()

        self.dm = dm
        self.solver = solver
        self.criterion = self.dm.criterion
        self.output_dir = engine.to_experiment_dir('outputs', self.dm.name)
        os.makedirs(self.output_dir, exist_ok=True)

    def validation_step(self, batch, batch_idx):

        outputs = self.solver(batch)
        preds = outputs.detach().cpu().numpy()
        B, C = preds.shape
        labels = batch['mos'].view(B, C)
        labels = labels.detach().cpu().numpy()
        
        self.plcc.update(preds, labels)
        self.srocc.update(preds, labels)
        self.krocc.update(preds, labels)
        self.rmse.update(preds, labels)

        # self._all_preds.append(preds)
        # self._all_labels.append(labels)
#        _, _, preds_normalized = self._logistic_4_fitting(preds)
#        labels_normalized = self._logistic_4_fitting(labels)
        # print(f'preds_norm: {preds_normalized[:20]}')
        # print(f'MOS_norm: {labels[:20]}')
       


    def on_validation_epoch_end(self):
        
        for i, crit in enumerate(self.criterion):
            plcc  = self.plcc.compute()   # 여기서 Lightning이 자동 all-gather
            srocc = self.srocc.compute()
            krocc = self.krocc.compute()
            rmse  = self.rmse.compute()

        # preds = np.concatenate(self._all_preds, axis=0)
        # labels = np.concatenate(self._all_labels, axis=0)


        # scores_no_fitted = {c: {} for c in self.criterion}
        # scores_fitted = {c: {} for c in self.criterion}
        # for i, crit in enumerate(self.criterion):
        #     pred = preds[:, i]
        #     gt = labels[:, i]
        #     print('pred: ', pred)
        #     print('labels:', gt)

        #     pred_norm = self._min_max_normalize(pred)
        #     gt_norm = self._min_max_normalize(gt)
        #     pred_norm_t = torch.from_numpy(pred_norm).to(self.device)
        #     gt_norm_t = torch.from_numpy(gt_norm).to(self.device)
            
        #     metrics_no_fitted = self._evaluate_metrics(pred_norm_t, gt_norm_t)
        #     scores_no_fitted[crit] = metrics_no_fitted

        #     _, _, pred_fitted = self._logistic_4_fitting(pred, gt)
        #     preds_t = torch.from_numpy(pred_fitted).to(self.device)
        #     gt_t = torch.from_numpy(gt).to(self.device)
            
        #     metrics_fitted = self._evaluate_metrics(preds_t, gt_t)
        #     scores_fitted[crit] = metrics_fitted
        
        result_str_no_fitted = ''
        result_str_fitted = ''
        for c in self.criterion:
            result_str_no_fitted += f"[{c}] PLCC: {scores_no_fitted[c]['plcc']}, SROCC: {scores_no_fitted[c]['srocc']}, KROCC: {scores_no_fitted[c]['krocc']}, RMSE: {scores_no_fitted[c]['rmse']}\n"
            result_str_fitted += f"[{c}] PLCC: {scores_fitted[c]['plcc']}, SROCC: {scores_fitted[c]['srocc']}, KROCC: {scores_fitted[c]['krocc']}, RMSE: {scores_fitted[c]['rmse']}\n"
        
        result = f'---No fitted Result---\n {result_str_no_fitted}' + '\n' + f'---fitted Result---\n {result_str_fitted}'
        print(result)
        out_path = osp.join(self.out_dir, f'eval_results_{self.criterion}.txt')
        
        with open(out_path, 'w') as f:
            f.write(result)


