import torch
from torch.nn import functional as F
from torchmetrics import Metric

class PLCC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_xy",  default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x",   default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y",   default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x2",  default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y2",  default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n",       default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        x = preds.detach().view(-1).double()
        y = target.detach().view(-1).double()

        self.sum_xy += torch.sum(x * y)
        self.sum_x += torch.sum(x)
        self.sum_y += torch.sum(y)
        self.sum_x2 += torch.sum(x * x)
        self.sum_y2 += torch.sum(y * y)
        self.n += x.numel()

    def compute(self):
        n = self.n
        sx = self.sum_x
        sy = self.sum_y
        sxy = self.sum_xy
        sx2 = self.sum_x2
        sy2 = self.sum_y2

        num = n * sxy - sx * sy
        den = torch.sqrt((n * sx2 - sx*sx) * (n * sy2 - sy*sy))

        return (num / den).float()