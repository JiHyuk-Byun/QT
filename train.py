
import os
from os import path as osp
from argparse import ArgumentParser
os.environ['NCCL_ASYNC_ERROR_HANDLING']='1'

from omegaconf import OmegaConf
import lightning.pytorch as pl

import engine

parser = ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True)
parser.add_argument('--ckpt_path', '-p', type=str, default=None, help='Path to the saved checkpoint.')
parser.add_argument('--exp_name', '-en', type=str, help='Name of the experiment')
parser.add_argument('--debug', '-d', action='store_true', default=False, help='debug mode (for sanity check)')
parser.add_argument('--gpus', '-g', default='-1',
                    help='GPU to use (num. GPU or gpu ids, follow pytorch-lightning convention). e.g., "-1" (all), "2" (2 GPU), "0,1" (GPU id 0, 1), "[0]" (GPU id 0)')
args = parser.parse_args()


def main():
    # Step 1. Get config argument
    cfg = engine.load_config(args.config)
    pl.seed_everything(cfg.get('seed', 123456))
    if args.debug:
        args.exp_name = 'debugging'

    # Step 2. Create output directory and save current configuration.
    engine.create_experiment_context(cfg.get('output_dir', None), args.exp_name)
    with open(osp.join(engine.to_experiment_dir('config.yaml')), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    model = engine.instantiate(cfg.model)
    dm = engine.instantiate(cfg.data)
    solver = engine.instantiate(cfg.solver, dm=dm, model=model)

    if args.ckpt_path is not None:
        solver.load_from_checkpoint(args.ckpt_path, strict=True)
        
    if args.debug:
        dm.enable_debug_mode()


    trainer: pl.Trainer = engine.prepare_trainer(cfg, gpus=args.gpus, debug=args.debug)
    trainer.fit(model=solver, datamodule=dm)


if __name__ == '__main__':
    main()
