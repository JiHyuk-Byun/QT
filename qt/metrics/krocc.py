import torch
from torch.nn import functional as F
from torchmetrics import Metric
import numpy as np
from scipy.stats import kendalltau

class KROCC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds",  default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds, target):
        self.preds.append(preds.detach().cpu().numpy().ravel())
        self.target.append(target.detach().cpu().numpy().ravel())

    def compute(self):
        preds = np.concatenate(self.preds, axis=0)
        target = np.concatenate(self.target, axis=0)
        tau, _ = kendalltau(preds, target)
        return torch.tensor(tau, dtype=torch.float32)