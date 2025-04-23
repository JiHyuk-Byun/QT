from os import path as osp
from typing import Any, List, Dict
import importlib

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from abc import ABC, abstractmethod

class QA3DBaseDataModule(LightningDataModule, ABC):

    def __init__(self,
                 name: str,
                 root_dir: str,
                 criterion: int,
                 batch_size: int,
                 num_workers: int,
                 eval_batch_size: int = -1):
        super().__init__()

        self.name = name
        self.root_dir = root_dir
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        dataset = self._get_dataset(is_train=True)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        batch_size = self.eval_batch_size if self.eval_batch_size > 0 else self.batch_size
        dataset = self._get_dataset(is_train=False)
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    @abstractmethod
    def _get_dataset(self, is_train: bool = True):
        pass
    
    @abstractmethod
    def collate_fn(self, items):
        pass
    