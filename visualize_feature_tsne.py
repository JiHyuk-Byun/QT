"""
Visualize Point-Transformer-V3 encoder features
(Objaverse vs Generated 3D point clouds)

▪ forward-hook 로 encoder 출력을 저장
▪ mean-pool → 512-d feature
▪ PCA(50) → t-SNE(2) 차원 축소
▪ matplotlib 산점도
--------------------------------------------------
> python vis_feats_tsne.py --ckpt path/to/full.ckpt
"""

import argparse, os, torch
from os import path as osp
from tqdm import tqdm
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import lightning.pytorch as pl
import torch_scatter

import engine
from qt.data import QA3DBaseDataModule, ObjaverseDataModule, GC3DDataModule
from qt.solvers import BaseSolver, EvaluationSolver


class FeatEvalSolver(EvaluationSolver):
    """EvaluationSolver + encoder feature bank"""

    def __init__(self, dm, solver, tag: str,
                 enable_metrics=False):
        super().__init__(dm, solver)
        self.feat_bank: List[torch.Tensor] = []
        self.tag = tag
        self.output_name = engine.to_experiment_dir('outputs', tag)
        self.enable_metrics = enable_metrics
        # ── forward-hook 한 줄로 encoder 출력 잡기
        self._enc_handle = solver.model.enc.register_forward_hook(
            lambda m, i, o: self.feat_bank.append(torch_scatter.scatter_mean(o.feat, o.batch, dim=0).detach().cpu())
        )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        self.feat_bank.clear()                # reset

    def on_validation_epoch_end(self):
        if self.enable_metrics:
            super().on_validation_epoch_end()     # 기존 metric 출력
        if self.global_rank != 0:
            return
        feats = torch.cat(self.feat_bank, dim=0)   # (N,512)
        torch.save(feats, f"{self.output_name}_feats.pt")
        print(f"saved {self.tag}_feats.pt  shape={feats.shape}")
        #self.visualize_tsne(feats) 

    def teardown(self, stage):
        # hook 해제 (안 하면 메모리 누수 가능)
        self._enc_handle.remove()

parser = argparse.ArgumentParser()
parser.add_argument('experiment_dir', type=str, help='path to the saved experiment dir')
parser.add_argument('--use_existing', '-f', action='store_true', help='decide whether use existing feature' )
parser.add_argument('--enable_metrics', '-m', action='store_true')
parser.add_argument('--gc3d', type=str, default='3dgc_test')
parser.add_argument('--objaverse', type=str, default='objaverse_multi_predict6_full')
parser.add_argument('--userstudy', type=str, default='userstudy_100_multipredict')
#parser.add_argument('--data', '-dm', type=str, default='userstudy_100', help='evaluation data module config path')
parser.add_argument('--exp_name', '-n', type=str, default=None, help='Name of the experiment')
#parser.add_argument("--ckpt", required=True, help="Lightning checkpoint (.ckpt)")
parser.add_argument('--output', '-o', type=str, default='tsne_domains')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--pca_dim", type=int, default=50)
parser.add_argument("--perplexity", type=float, default=30)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

# ──────────────────────────────────────────────
# 2. 모델 로드 (LitModel 예시) & hook 설정
# ──────────────────────────────────────────────

engine.set_context_from_existing(args.experiment_dir)
cfg = engine.load_config(engine.to_experiment_dir('config.yaml'))
gc3d_cfg = engine.load_config(osp.join('config/data', f'{args.gc3d}.yaml'))
vlm_cfg = engine.load_config(osp.join('config/data', f'{args.objaverse}.yaml'))
userstudy_cfg = engine.load_config(osp.join('config/data', f'{args.userstudy}.yaml'))
pl.seed_everything(cfg.get('seed', 123456))

model = engine.instantiate(cfg.model)
dm: QA3DBaseDataModule = engine.instantiate(vlm_cfg)
solver: BaseSolver = engine.instantiate(cfg.solver, dm=dm, model=model)

best_ckpt_path = engine.find_best_checkpoint_path(cfg)
solver.load_checkpoint(best_ckpt_path)

solver.eval().to(args.device)

# ──────────────────────────────────────────────
# 3. 데이터 로더 준비
# ──────────────────────────────────────────────

domains = [
    ("objaverse/vlm_train",  vlm_cfg, "train"),   # cfg, split tag
    ("objaverse/vlm_val", vlm_cfg, "validate"),
    ("objaverse/userstudy", userstudy_cfg, "validate"),
    ("gc3d", gc3d_cfg, "validate"),
]

for tag, cfg_dm, stage in domains:
    dm = engine.instantiate(cfg_dm)
    dm.name = tag                                 # DM 내부 name 덮어쓰기
    dm.setup(stage=stage)

    feat_eval = FeatEvalSolver(dm, solver, tag, enable_metrics= args.enable_metrics)   # tag 넘김
    trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False)
    loader = dm.train_dataloader() if stage == "train" else dm.val_dataloader()
    print(f'{tag} size: {len(loader.dataset)}')
    trainer.validate(feat_eval, dataloaders=loader)

# ──────────────────────────────────────────────
# 4. Visualization
# ──────────────────────────────────────────────
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch, numpy as np, os

out_root = engine.to_experiment_dir("outputs")
tags = ["objaverse/vlm_train", "objaverse/vlm_val", "objaverse/userstudy", "gc3d"]

feat_list, label_list = [], []
for i, tag in enumerate(tags):
    f = torch.load(os.path.join(out_root, f"{tag}_feats.pt"))
    feat_list.append(f)
    label_list.append(np.full(len(f), i))      # domain id

X = torch.vstack(feat_list).numpy()
y = np.hstack(label_list)

# ▸ 차원 축소
X50 = PCA(50).fit_transform(X)
X2  = TSNE(2, perplexity=args.perplexity, init="pca").fit_transform(X50)

# ▸ 시각화
colors = cm.get_cmap("tab10").colors          # 10가지 색 팔레트
plt.figure(figsize=(7, 6))
for i, tag in enumerate(tags):
    plt.scatter(X2[y==i, 0], X2[y==i, 1],
                s=6, alpha=0.5, color=colors[i], label=tag)
plt.legend(frameon=False); plt.axis("off"); plt.tight_layout()

out_png = osp.join(out_root, f"{args.output}.png")
plt.savefig(out_png, dpi=300); plt.close()
print(f"t-SNE saved → {out_png}")