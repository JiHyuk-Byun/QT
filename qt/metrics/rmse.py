import torch
from torchmetrics import Metric

_EPS = 1e-8  


class RMSE(Metric):
    full_state_update: bool = False  
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n",   default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        err = (preds.detach() - target.detach()).double()
        self.sse += torch.sum(err ** 2)
        self.n   += err.numel()

    def compute(self) -> torch.Tensor:
        return torch.sqrt(self.sse / (self.n + _EPS)).float()