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
    def __init__(self, dm, solver, tag: str, enable_metrics=False, crit_idx=5):
        super().__init__(dm, solver)
        self.tag, self.enable_metrics = tag, enable_metrics
        self.output_name = engine.to_experiment_dir('outputs', tag)
        self.crit_idx = crit_idx            # MOS column to pick
        self.feat_bank, self.score_bank = [], []

        # encoder hook: pooled feature
        self._enc_handle = solver.model.enc.register_forward_hook(
            lambda m, i, o:
                self.feat_bank.append(torch_scatter.scatter_mean(o.feat, o.batch, dim=0).cpu())
        )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        self.feat_bank.clear()                # reset
    
    def validation_step(self, batch, batch_idx):
        
        _ = self.solver(batch)          # forward만 (pred 무시)
        labels = batch['mos'].view(-1, 6)

        self.score_bank.append(labels[:, self.crit_idx].cpu())  # (B,)

    def on_validation_epoch_end(self):
        if self.enable_metrics:
            super().on_validation_epoch_end()

        if self.global_rank != 0: return
        feats  = torch.cat(self.feat_bank,  dim=0)
        scores = torch.cat(self.score_bank, dim=0)   # (N,)
        s_min, s_max = scores.min(), scores.max()
        scores = (scores - s_min) / (s_max - s_min) * 4 + 1
        
        torch.save({'feat': feats, 'score': scores},
                   f"{self.output_name}_feats.pt")
        print(f"saved {self.tag}_feats.pt  shape={tuple(feats.shape)}")

    def teardown(self, stage): self._enc_handle.remove()

parser = argparse.ArgumentParser()
parser.add_argument('experiment_dir', type=str, help='path to the saved experiment dir')
parser.add_argument('--use_existing', '-u', action='store_true', help='decide whether use existing feature' )
parser.add_argument('--enable_metrics', '-m', action='store_true')

parser.add_argument('--gc3d', type=str, default='3dgc_test')
parser.add_argument('--objaverse', type=str, default='objaverse_multi_predict6_full')
parser.add_argument('--userstudy', type=str, default='userstudy_100_multipredict')

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
#    ("objaverse/vlm_train",  vlm_cfg, "train"),   # cfg, split tag
    ("objaverse/vlm_val", vlm_cfg, "validate"),
    ("objaverse/userstudy", userstudy_cfg, "validate"),
    ("gc3d", gc3d_cfg, "validate"),
]

for tag, cfg_dm, stage in domains:
    print(f'{tag} processing...')
    dm = engine.instantiate(cfg_dm)
    dm.name = tag                                 # DM 내부 name 덮어쓰기
    dm.setup(stage=stage)

    feat_eval = FeatEvalSolver(dm, solver, tag, enable_metrics= args.enable_metrics)   # tag 넘김
    trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False)
    loader = dm.train_dataloader() if stage == "train" else dm.val_dataloader()
    print(f'size: {len(loader.dataset)}')
    trainer.validate(feat_eval, dataloaders=loader)

# ──────────────────────────────────────────────
# 4. Visualization
# ──────────────────────────────────────────────
import matplotlib.pyplot as plt, numpy as np, torch, os
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tags = [
    #"objaverse/vlm_train", 
    "objaverse/vlm_val", 
    "objaverse/userstudy", 
    "gc3d"]
markers = ["o", "s", "^", "X"]          # 4 domains
score_colors = cm.get_cmap("RdYlBu_r", 5)  # 5 discrete colours (1 ~ 5)

feat_all, dom_all, score_all = [], [], []
for d_idx, tag in enumerate(tags):
    pack = torch.load(os.path.join(out_root, f"{tag}_feats.pt"))
    feat_all.append(pack['feat'])
    dom_all.append(np.full(len(pack['feat']), d_idx))
    score_all.append(pack['score'].numpy())   # assume 1 ~ 5 already
X   = torch.vstack(feat_all).numpy()
dom = np.hstack(dom_all)
scr = np.hstack(score_all)          # float → int
scr = np.clip(np.round(scr).astype(int), 1, 5)

# ── 차원 축소
X50 = PCA(50).fit_transform(X)
X2  = TSNE(2, perplexity=args.perplexity, init="pca").fit_transform(X50)

# ── 시각화: domain 별 marker, score 별 color
plt.figure(figsize=(7,6))
for d_idx, tag in enumerate(tags):
    for s in range(1,6):
        mask = (dom==d_idx) & (scr==s)
        if mask.any():
            plt.scatter(X2[mask,0], X2[mask,1],
                        s=6, marker=markers[d_idx],
                        color=score_colors(s-1), alpha=.6,
                        label=f"{tag}-score{s}" if d_idx==0 else None)

# 범례: 첫 도메인만 색설명, markers는 개별 텍스트로 추가
from matplotlib.lines import Line2D
color_handles = [Line2D([0],[0], marker='o', color=score_colors(i),
                        linestyle='None', markersize=6, label=f"score {i+1}")
                 for i in range(5)]
marker_handles = [Line2D([0],[0], marker=markers[i], color='k',
                         linestyle='None', markersize=7, label=tags[i])
                  for i in range(len(tags))]
plt.legend(handles=color_handles+marker_handles, frameon=False, ncol=2)

plt.axis('off'); plt.tight_layout()
plt.savefig(os.path.join(out_root, f"{args.output}_score_colored.png"), dpi=300)
print("t-SNE saved with score colouring.")