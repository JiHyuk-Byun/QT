import torch
from torch.nn import functional as F
from torchmetrics import Metric
import numpy as np
from scipy.stats import spearmanr

class SROCC(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # we store lists of numpy arrays; no distributed reduction
        self.add_state("preds",  default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # detach, move to CPU, convert to numpy and flatten
        self.preds.append(preds.detach().cpu().numpy().ravel())
        self.target.append(target.detach().cpu().numpy().ravel())

    def compute(self):

        preds = np.concatenate(self.preds, axis=0)
        target = np.concatenate(self.target, axis=0)
        corr, _ = spearmanr(preds, target)
        # wrap back into torch tensor
        return torch.tensor(corr, dtype=torch.float32)