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

    def __init__(self, dm, solver):
        super().__init__(dm, solver)
        self.feat_bank: List[torch.Tensor] = []

        # ── forward-hook 한 줄로 encoder 출력 잡기
        self._enc_handle = solver.model.enc.register_forward_hook(
            lambda m, i, o: self.feat_bank.append(torch_scatter.scatter_mean(o.feat, o.batch, dim=0).detach().cpu())
        )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        self.feat_bank.clear()                # reset

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()     # 기존 metric 출력

        if self.global_rank == 0:
            feats = torch.cat(self.feat_bank, dim=0)   # (N,512)
            domain = self.dm.name                      # 'objaverse' / 'gc3d'
            torch.save(feats, f"{self.output_dir}/{domain}_feats.pt")
            print(f"saved {domain}_feats.pt  shape={feats.shape}")
            #self.visualize_tsne(feats) 

    def teardown(self, stage):
        # hook 해제 (안 하면 메모리 누수 가능)
        self._enc_handle.remove()

    def visualize_tsne(self, feats: torch.Tensor):
        """
        feats : (N,512) encoder pooled feature
        저장 경로 : {output_dir}/{dm.name}_tsne.png
        """
        X = feats.numpy()
        # ▸ 선행 PCA → 속도 안정화
        X50 = PCA(50, random_state=0).fit_transform(X)
        X2  = TSNE(
                n_components=2, perplexity=30,
                init="pca", random_state=0
              ).fit_transform(X50)

        plt.figure(figsize=(6, 6))
        plt.scatter(X2[:, 0], X2[:, 1], s=6, alpha=.5)
        plt.title(f"t-SNE – {self.dm.name}")
        plt.axis("off"); plt.tight_layout()

        png_path = os.path.join(self.output_dir, f"{self.dm.name}_tsne.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"t-SNE saved → {png_path}")

parser = argparse.ArgumentParser()
parser.add_argument('experiment_dir', type=str, help='path to the saved experiment dir')
parser.add_argument('--gc3d', type=str, default='3dgc_test')
parser.add_argument('--objaverse', type=str, default='userstudy_100_multipredict')
#parser.add_argument('--data', '-dm', type=str, default='userstudy_100', help='evaluation data module config path')
parser.add_argument('--exp_name', '-n', type=str, default=None, help='Name of the experiment')
#parser.add_argument("--ckpt", required=True, help="Lightning checkpoint (.ckpt)")
parser.add_argument('--output', '-o', type=str, default='tsne_real_vs_gen')
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
objaverse_cfg = engine.load_config(osp.join('config/data', f'{args.objaverse}.yaml'))
pl.seed_everything(cfg.get('seed', 123456))

model = engine.instantiate(cfg.model)
dm: QA3DBaseDataModule = engine.instantiate(objaverse_cfg)
solver: BaseSolver = engine.instantiate(cfg.solver, dm=dm, model=model)

best_ckpt_path = engine.find_best_checkpoint_path(cfg)
solver.load_checkpoint(best_ckpt_path)

solver.eval().to(args.device)

# ──────────────────────────────────────────────
# 3. 데이터 로더 준비
# ──────────────────────────────────────────────

dm_real = engine.instantiate(objaverse_cfg)
dm_synth = engine.instantiate(gc3d_cfg)
print("Objaverse eval size :", len(dm_real.val_dataloader().dataset))
print("GC3D eval size      :", len(dm_synth.val_dataloader().dataset))

for dm in [dm_real, dm_synth]:
    dm.setup(stage='validate')
    feat_eval = FeatEvalSolver(dm, solver)
    trainer = pl.Trainer(accelerator='gpu', devices=[0], logger=False)
    trainer.validate(feat_eval, dataloaders=dm.val_dataloader())


import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

f_real  = torch.load(f"{engine.to_experiment_dir('outputs')}/objaverse/objaverse_feats.pt")
f_synth = torch.load(f"{engine.to_experiment_dir('outputs')}/gc3d/gc3d_feats.pt")

X = torch.vstack([f_real, f_synth]).numpy()
y = np.hstack([np.zeros(len(f_real)), np.ones(len(f_synth))])

X50 = PCA(50).fit_transform(X)
X2  = TSNE(2, perplexity=30, init="pca").fit_transform(X50)

plt.scatter(X2[y==0,0], X2[y==0,1], s=6, alpha=.5, label="Objaverse")
plt.scatter(X2[y==1,0], X2[y==1,1], s=6, alpha=.5, label="Generated")
plt.legend(); plt.axis("off"); plt.tight_layout()
plt.savefig(f"{engine.to_experiment_dir('outputs')}/{args.output}.png", dpi=300)