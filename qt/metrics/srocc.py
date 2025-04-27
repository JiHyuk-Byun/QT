import torch
from torchmetrics import Metric
from scipy.stats import spearmanr


class SROCC(Metric):
    """
    Spearman Rank-Order Correlation Coefficient (ρ)
    • scipy.stats.spearmanr(preds, targets) 사용
    • preds, targets: shape (N, …)  →  1-D 로 평탄화한 뒤 계산
    • 반환: torch.tensor(float)  (ρ 값 하나)
    """
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # 각 프로세스(DDP)에서 모은 결과를 cat 으로 합칩니다.
        self.add_state("preds",
                       default=torch.empty(0, dtype=torch.float32),
                       dist_reduce_fx="cat")
        self.add_state("targets",
                       default=torch.empty(0, dtype=torch.float32),
                       dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        각 step마다 예측·정답을 모아서 1-D CPU 텐서에 누적.
        """
        preds_1d   = preds.detach().view(-1).cpu()
        targets_1d = targets.detach().view(-1).cpu()

        self.preds   = torch.cat([self.preds, preds_1d])
        self.targets = torch.cat([self.targets, targets_1d])

    def compute(self) -> torch.Tensor:
        """
        모든 step에서 누적한 데이터를 이용해 SROCC 산출.
        """
        if self.preds.numel() == 0:
            return torch.tensor(float("nan"))

        rho, _ = spearmanr(self.preds.numpy(), self.targets.numpy())
        return torch.tensor(rho, dtype=torch.float32)