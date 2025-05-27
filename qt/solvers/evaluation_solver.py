import os
from os import path as osp
from typing import List
import csv

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
        preds = outputs.detach()
        # print('#'*100)
        # print('This is validation step')
        # for k, v in batch.items():
        #     print('-'*40)
        #     print(f'key: {k}')
        #     print(f'value:{v}')
            
        #     if isinstance(v, torch.Tensor):
        #         print(f'shape: {v.shape}')
        #         print(f'scale: {v.min()}~{v.max()}')
        #     elif isinstance(v, list):
        #         print(f'len: {len(v)}')
        #         print(f'scale: {min(v)}~{max(v)}')
        #     else:
        #         print(f'TypeError: {v.type}')
                

        ids = batch['id']
        labels = batch['mos'].view_as(preds)
        labels = labels.detach()
        
        
        self._all_preds.append(preds)
        self._all_labels.append(labels)
        self._all_ids.append(ids)
#        _, _, preds_normalized = self._logistic_4_fitting(preds)
#        labels_normalized = self._logistic_4_fitting(labels)
        # print(f'preds_norm: {preds_normalized[:20]}')
        # print(f'MOS_norm: {labels[:20]}')
       


    def on_validation_epoch_end(self):
        
        #for i, crit in enumerate(self.criterion):

        preds_local = torch.cat(self._all_preds, dim=0)
        labels_local = torch.cat(self._all_labels, dim=0)
        ids_local = self._all_ids

        preds_all = self.all_gather(preds_local)
        labels_all = self.all_gather(labels_local)
        ids_all = self.all_gather(ids_local)

        if self.global_rank != 0:
            return
        assert len(self.criterion) == preds_all.shape[-1]

        preds_all = preds_all.reshape(-1, len(self.criterion)).cpu()
        labels_all = labels_all.reshape(-1, len(self.criterion)).cpu()
        ids_all = [id for sub in ids_all for id in sub]

        self._save_csv(ids_all, preds_all, labels_all)
        
        scores_no_fitted = dict()
        scores_fitted = dict()
        for i, crit in enumerate(self.criterion):
            pred = preds_all[:, i].numpy()
            gt = labels_all[:, i].numpy()
            print('pred: ', pred)
            print('labels:', gt)

            pred_norm = self._min_max_normalize(pred)
            gt_norm = self._min_max_normalize(gt)
            pred_norm_t = torch.from_numpy(pred_norm)
            gt_norm_t = torch.from_numpy(gt_norm)
            
            metrics_no_fitted = self._evaluate_metrics(pred_norm_t, gt_norm_t)
            scores_no_fitted[crit] = metrics_no_fitted

            _, _, pred_fitted = self._logistic_4_fitting(pred, gt)
            preds_t = torch.from_numpy(pred_fitted)
            gt_t = torch.from_numpy(gt)
            
            metrics_fitted = self._evaluate_metrics(preds_t, gt_t)
            scores_fitted[crit] = metrics_fitted
        
        result_str_no_fitted = ''
        result_str_fitted = ''
        for c in self.criterion:
            result_str_no_fitted += f"[{c}] PLCC: {scores_no_fitted[c]['plcc']}, SROCC: {scores_no_fitted[c]['srocc']}, KROCC: {scores_no_fitted[c]['krocc']}, RMSE: {scores_no_fitted[c]['rmse']}\n"
            result_str_fitted += f"[{c}] PLCC: {scores_fitted[c]['plcc']}, SROCC: {scores_fitted[c]['srocc']}, KROCC: {scores_fitted[c]['krocc']}, RMSE: {scores_fitted[c]['rmse']}\n"
        
        result = f'---No fitted Result---\n {result_str_no_fitted}' + '\n' + f'---fitted Result---\n {result_str_fitted}'
        print(result)
        out_path = osp.join(self.output_dir, f'eval_results_{self.criterion}.txt')
        
        with open(out_path, 'w') as f:
            f.write(result)


def _save_csv(self, ids: List[str], preds: torch.Tensor, scores: torch.Tensor):

    eval_result = {'id': ids}                          
    for i, crit in enumerate(self.criterion):          
        eval_result[crit] = preds[:, i].tolist()

    eval_result['gt(preference)'] = (
        scores[:, 5].tolist() if scores.dim() == 2 else scores.tolist()
    )

    outpath = osp.join(self.output_dir, "eval_scores.csv")

    header = list(eval_result.keys())                  # ['id', crit1, crit2, ..., 'gt(preference)']
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row_vals in zip(*eval_result.values()):
            writer.writerow(dict(zip(header, row_vals)))

