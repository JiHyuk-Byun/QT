from os import path as osp
from typing import Dict
import importlib
import io, time

import torch
from torch.utils.data import Dataset
import numpy as np

from .base import QA3DBaseDataModule
from .utils import read_csv, pc_normalize, to_tensor, shuffle_point, GridSample, Collect

CRITERIA = {
    "geometry": 0,
    "texture": 1,
    "material": 2,
    "plausibility": 3,
    "artifact": 4,
    "preference": 5
}

MAX_RETRY = 3  
RETRY_SLEEP = 0.1 

class ObjaverseDataModule(QA3DBaseDataModule):

    def __init__(self, 
                 root_dir: str,
                 train_split: str,
                 test_split: str,
                 criterion: str,
                 batch_size: int,
                 num_workers: int,
                 dataset_config: dict,
                 eval_batch_size: int = -1,
                 ):
        super().__init__('objaverse',
                         root_dir,
                         train_split,
                         test_split,
                         criterion,
                         batch_size,
                         num_workers,
                         eval_batch_size)

        self.dataset_config = dataset_config
    
    def _get_dataset(self, is_train: bool = True):
        split = 'train' if is_train else 'test'
        return ObjaverseDataset(**self.dataset_config, root_dir=self.root_dir, train_split=self.train_split, test_split=self.test_split, criterion=self.criterion, split=split)
    
class ObjaverseDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 train_split: str,
                 test_split: str,
                 criterion: str,
                 split: str,
                 augments: Dict[str, dict],

                 grid_size: int = 0.02,
                 hash_type: str = 'fnv',
                 return_grid_coord: bool = True,

                 keys=['coord', 'grid_coord', 'mos'],
                 feat_keys=['coord', 'normal'],
                 manual_seed: int = None):
        
        super().__init__()
        self.root_dir = root_dir

        assert split in ['train', 'test']
        self.split = split
        self.train_split = osp.join(root_dir, 'train_split.csv') if train_split is None else train_split
        self.test_split = osp.join(root_dir, 'test_split.csv') if test_split is None else test_split
        
        self.split_path = self.train_split if split == 'train' else self.test_split
        
        self.criterion = criterion
        self.criterion_idx = CRITERIA[self.criterion]
        self.manual_seed = manual_seed

        self.files = read_csv(self.split_path)
        
        #Augmentation
        self.augment_fns = self._compose(augments)

        self.grid_sampler = GridSample(grid_size=grid_size, hash_type=hash_type, return_grid_coord=return_grid_coord, mode='train')
        
        self.collect_keys = Collect(keys=keys, feat_keys=feat_keys)
        # after collect, e.g. key: [coord, grid_coord, 'mos', 'offset', 'feat'(which is concatnated)] 
    
    def _compose(self, augments: Dict[str, dict]):
        
        transform_fns = []
        if augments == None:
            return transform_fns

        utils = importlib.import_module('qt.data.utils')
        for fn_name, args in augments.items():
            if hasattr(utils, fn_name):
                transform_fns.append(getattr(utils, fn_name)(**args))
            else:
                raise AttributeError(f'No function named {fn_name} defined in utils.py')

        return transform_fns
    
    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        file_path = osp.join(self.root_dir, self.files[idx][0])
        MOSlabels = self.files[idx][1:]

        for attempt in range(MAX_RETRY):
            try:
                data = np.load(file_path, allow_pickle=True).item()
                break                                
            except Exception as e:
                if attempt < MAX_RETRY - 1:
                    time.sleep(RETRY_SLEEP)          
                else:

                    if (not torch.distributed.is_initialized()) \
                       or torch.distributed.get_rank() == 0:
                        print(f"[SKIP] idx={idx} path={file_path} "
                              f"({type(e).__name__}: {e})")
                    return None


        data_dict = {
            'coord' : data['coord'].astype(np.float32),
            'color' : data['color'].astype(np.float32),
            'normal': data['normal'].astype(np.float32),
            'mos'   : torch.tensor([MOSlabels[self.criterion_idx]],
                                   dtype=torch.float32),
        }

        data_dict = pc_normalize(data_dict)
        #Augmentation
        for fn in self.augment_fns:
            data_dict = fn(data_dict)
        # Grid sampling
        data_dict = self.grid_sampler(data_dict)
        data_dict = to_tensor(shuffle_point(data_dict))
        return self.collect_keys(data_dict)