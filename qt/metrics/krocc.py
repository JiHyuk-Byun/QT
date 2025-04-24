import torch
from torchmetrics import Metric

_EPS = 1e-8


class KROCC(Metric):
    r"""Kendall Rank-Order Correlation Coefficient (τ-b).

    Works with arbitrary batch sizes; handles ties via τ-b formulation.
    """
    full_state_update: bool = False  # 한 배치마다 accumulate → 메모리 절약

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # 상태값 누적 ― 모두 double precision
        self.add_state("concordant", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("discordant", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ties_x",     default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ties_y",     default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Accumulate pair statistics for a batch."""
        x = preds.detach().view(-1).double()
        y = target.detach().view(-1).double()
        if x.numel() <= 1:                       
            return

        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]

        mask = torch.triu(torch.ones_like(diff_x, dtype=torch.bool), diagonal=1)
        dx = diff_x[mask]
        dy = diff_y[mask]
        prod = dx * dy

        self.concordant += (prod > 0).sum().double()
        self.discordant += (prod < 0).sum().double()
        self.ties_x     += (dx == 0).sum().double()
        self.ties_y     += (dy == 0).sum().double()

    def compute(self) -> torch.Tensor:
        C  = self.concordant
        D  = self.discordant
        Tx = self.ties_x
        Ty = self.ties_y

        denom = torch.sqrt((C + D + Tx) * (C + D + Ty) + _EPS)
        return ((C - D) / denom).float()