import torch
from torchmetrics import Metric
from scipy.stats import kendalltau


class KROCC(Metric):
    """
    Kendall Rank-Order Correlation Coefficient (τ-b)
    • scipy.stats.kendalltau(preds, targets, variant="b") 사용
    • preds / targets: shape (N, …) ― 어떤 차원이든 OK, 1-D로 펼쳐서 계산
    • 반환값: torch.tensor(float)  ⟵  τ 값만 사용
    """
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds",   default=torch.empty(0, dtype=torch.float32),
                       dist_reduce_fx="cat")
        self.add_state("targets", default=torch.empty(0, dtype=torch.float32),
                       dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # 모든 차원을 1-D로
        preds_1d   = preds.detach().view(-1).cpu()
        targets_1d = targets.detach().view(-1).cpu()

        self.preds   = torch.cat([self.preds, preds_1d])
        self.targets = torch.cat([self.targets, targets_1d])

    def compute(self) -> torch.Tensor:
        if self.preds.numel() == 0:
            return torch.tensor(float("nan"))

        tau, _ = kendalltau(self.preds.numpy(), self.targets.numpy(),
                            variant="b")  # ties 처리
        return torch.tensor(tau, dtype=torch.float32)