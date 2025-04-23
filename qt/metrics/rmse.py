import torch
from torch.nn import functional as F
from torchmetrics import Metric


class RMSE(Metric):
    """
    Root Mean Squared Error (RMSE) metric implemented in PyTorch without SciPy.
    Accumulates sum of squared errors and count over batches, then computes sqrt(MSE).
    """
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # state variables for distributed reduction
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n",                   default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Called on every batch. Add batch's squared error and element count.
        """
        x = preds.detach().view(-1).double()
        y = target.detach().view(-1).double()
        self.sum_squared_error += torch.sum((x - y) ** 2)
        self.n += x.numel()

    def compute(self):
        """
        Computes and returns the RMSE over all batches.
        """
        mse = self.sum_squared_error / self.n
        return torch.sqrt(mse).float()