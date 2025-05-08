import random
from typing import List
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from abc import ABC, abstractmethod

CRITERIA = {
    "geometry": 0,
    "texture": 1,
    "material": 2,
    "plausibility": 3,
    "artifact": 4,
    "preference": 5
}


class QA3DBaseDataModule(LightningDataModule, ABC):

    def __init__(self,
                 name: str,
                 root_dir: str,
                 train_split: str,
                 test_split: str,
                 criterion: list,
                 batch_size: int,
                 num_workers: int,
                 eval_batch_size: int = -1):
        super().__init__()

        self.name = name
        self.root_dir = root_dir
        self.train_split = train_split
        self.test_split = test_split
        self.criterion = criterion
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size

    def setup(self, stage=None):
        pass

    def enable_debug_mode(self):
        self.batch_size = 1
        self.eval_batch_size = 1
        self.num_workers = 0

    def train_dataloader(self):
        dataset = self._get_dataset(is_train=True)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=self._collate_fn, drop_last=True, persistent_workers=False)

    def val_dataloader(self):
        batch_size = self.eval_batch_size if self.eval_batch_size > 0 else self.batch_size
        dataset = self._get_dataset(is_train=False)
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers,
                          collate_fn=self._collate_fn, drop_last=True, persistent_workers=False)

    @abstractmethod
    def _get_dataset(self, is_train: bool = True):
        pass

    def _collate_fn(self, batch):
        """
        collate function for point cloud which support dict and list,
        'coord' is necessary to determine 'offset'
        """

        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        if not isinstance(batch, Sequence):
            raise TypeError(f"{batch.dtype} is not supported.")

        if isinstance(batch[0], torch.Tensor):
            return torch.cat(list(batch))
        elif isinstance(batch[0], str):
            # str is also a kind of Sequence, judgement should before Sequence
            return list(batch)
        elif isinstance(batch[0], Sequence):
            for data in batch:
                data.append(torch.tensor([data[0].shape[0]]))
            batch = [self._collate_fn(samples) for samples in zip(*batch)]
            batch[-1] = torch.cumsum(batch[-1], dim=0).int()
            return batch
        elif isinstance(batch[0], Mapping):
            batch = {
                key: (
                    self._collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        self._collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                        dim=0,
                    )
                )
                for key in batch[0]
            }
            return batch
        else:
            return default_collate(batch)

    def _point_collate_fn(self, batch, mix_prob=0):
        assert isinstance(
            batch[0], Mapping
        )  # currently, only support input_dict, rather than input_list
        batch = self.collate_fn(batch)
        if random.random() < mix_prob:
            if "instance" in batch.keys():
                offset = batch["offset"]
                start = 0
                num_instance = 0
                for i in range(len(offset)):
                    if i % 2 == 0:
                        num_instance = max(batch["instance"][start: offset[i]])
                    if i % 2 != 0:
                        mask = batch["instance"][start: offset[i]] != -1
                        batch["instance"][start: offset[i]] += num_instance * mask
                    start = offset[i]
            if "offset" in batch.keys():
                batch["offset"] = torch.cat(
                    [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
                )
        return batch
