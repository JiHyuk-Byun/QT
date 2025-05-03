# test for userstudy

from os import path as osp
from argparse import ArgumentParser

import lightning.pytorch as pl

import engine
from qt.data import QA3DBaseDataModule
from qt.solvers import BaseSolver, EvaluationSolver


parser = ArgumentParser()
parser.add_argument('experiment_dir', type=str, help='path to the saved experiment dir')
parser.add_argument('--data', '-dm', type=str, default='userstudy_100', help='evaluation data module config path')
#parser.add_argument('--name', '-n', type=str, default=None, help='Name of the experiment')
parser.add_argument('--gpus', '-g', default='[1]',
                    help='GPU to use (num. GPU or gpu ids, follow pytorch-lightning convention). e.g., "-1" (all), "2" (2 GPU), "0,1" (GPU id 0, 1), "[0]" (GPU id 0)')


def main():
    args = parser.parse_args()
    engine.set_context_from_existing(args.experiment_dir)
    cfg = engine.load_config(engine.to_experiment_dir('config.yaml'))
    dm_cfg = engine.load_config(osp.join('config/data', f'{args.data}.yaml'))
    pl.seed_everything(cfg.get('seed', 123456))

    model = engine.instantiate( cfg.model)
    dm: QA3DBaseDataModule = engine.instantiate(dm_cfg)
    solver: BaseSolver = engine.instantiate(cfg.solver, dm=dm, model=model)

    best_ckpt_path = engine.find_best_checkpoint_path(cfg)
    solver.load_checkpoint(best_ckpt_path)
    # dm.preprocess()

    eval_solver = EvaluationSolver(dm=dm, solver=solver)
    gpus = engine.parse_gpus_str(args.gpus)
    trainer = pl.Trainer(devices=gpus)
    trainer.validate(model=eval_solver, datamodule=dm)


if __name__ == '__main__':
    main()
