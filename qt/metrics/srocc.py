import torch
from torchmetrics import Metric

class SROCC(Metric):

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # accumulate all preds and targets across batches
        self.add_state("preds",  default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with a new batch of preds and targets.
        """
        p = preds.detach().view(-1).double()
        t = target.detach().view(-1).double()
        self.preds   = torch.cat([self.preds,   p])
        self.target  = torch.cat([self.target,  t])

    def compute(self):
        """
        Compute the Spearman correlation over all accumulated data.
        """
        n = self.preds.numel()
        if n < 2:
            return torch.tensor(0.0, dtype=torch.float32)
        # rank data (0-based)
        rank_p = self.preds.argsort().argsort().double()
        rank_t = self.target.argsort().argsort().double()
        d = rank_p - rank_t
        sum_d2 = torch.sum(d * d)
        # Spearman formula (no tie correction)
        denom = n * (n**2 - 1)
        srocc = 1 - (6 * sum_d2) / denom
        return srocc.float()