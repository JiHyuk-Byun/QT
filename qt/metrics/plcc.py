import torch
from torchmetrics import Metric
from scipy.stats import pearsonr


class PLCC(Metric):
    """
    Pearson Linear Correlation Coefficient (PLCC)
    implemented with scipy.stats.pearsonr.

    • preds / targets: shape (N, …) ― 어떤 차원이든 OK, 일렬로 펴서 계산
    • 반환값: torch.tensor(float)  ⟵  pearsonr 의 r 값만 사용
    """
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        print(self.device)
        # GPU ↔︎ CPU 간 통신을 위해 텐서로 보관하고, cat 연산으로 집계
        self.add_state(
            "preds",
            default=torch.empty(0, dtype=torch.float32),
            dist_reduce_fx="cat"
        )
        self.add_state(
            "targets",
            default=torch.empty(0, dtype=torch.float32),
            dist_reduce_fx="cat"
        )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds, targets ── shape (B, …).  일렬로 펴서 CPU에 저장
        """
        preds_1d   = preds.detach().view(-1).cpu()
        targets_1d = targets.detach().view(-1).cpu()

        self.preds   = torch.cat([self.preds, preds_1d])
        self.targets = torch.cat([self.targets, targets_1d])

    def compute(self) -> torch.Tensor:
        """
        모든 step에서 모은 값을 바탕으로 PLCC 반환
        """
        if self.preds.numel() == 0:
            # 데이터가 없으면 NaN 반환
            return torch.tensor(float("nan"))

        r_value, _ = pearsonr(self.preds.numpy(), self.targets.numpy())
        return torch.tensor(r_value, dtype=torch.float32)