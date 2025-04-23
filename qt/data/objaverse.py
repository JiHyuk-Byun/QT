from os import path as osp
from typing import Any, List, Dict
import importlib

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np

from .base import QA3DBaseDataModule
from .base_dataset import BaseDataset
from .utils import read_csv, pc_normalize, to_tensor, shuffle_point, GridSample, Collect

class ObjaverseDataModule(QA3DBaseDataModule):

    def __init__(self, 
                 root_dir: str,
                 criterion: int,
                 batch_size: int,
                 num_workers: int,
                 eval_batch_size: int = -1,
                 **dataset_config):
        super().__init__('objaverse',
                         root_dir,
                         criterion,
                         batch_size,
                         num_workers,
                         eval_batch_size)

        self.dataset_config = dataset_config
    
    def _get_dataset(self, is_train: bool = True):
        split = 'train' if is_train else 'test'
        return ObjaverseDataset(self.dataset_config, root_dir=self.root_dir, criterion=self.criterion, split=split)

    def collate_fn(self, items):
        return items
    
class ObjaverseDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 criterion: int,
                 split: str,
                 augments: Dict[dict],

                 grid_size: int = 0.01,
                 hash_type: str = 'fnv',
                 return_grid_coord: bool = True,

                 keys=['coord', 'grid_coord', 'mos'],
                 feat_keys=['coord', 'normal'],
                 manual_seed: int = None):
        
        super().__init__(root_dir, criterion, split, manual_seed)

        self.files = read_csv(osp.join(self.root_dir, f'{self.split}_split.xlsx'))
        
        #Augmentation
        self.augment_fns = self._compose(augments)

        self.grid_sampler = GridSample(grid_size, hash_type, return_grid_coord, mode=split)
        
        self.collect_keys = Collect(keys=keys, feat_keys=feat_keys)
        # after collect, e.g. key: [coord, grid_coord, 'mos', 'offset', 'feat'(which is concatnated)] 
    
    def _compose(self, augments: Dict[dict]):
        transform_fns = []
        utils = importlib.import_module('utils')
        for fn_name, args in augments.items():
            if hasattr(utils, fn_name):
                transform_fns.append(getattr(utils, fn_name(**args)))
            else:
                raise AttributeError(f'No function named {fn_name} defined in utils.py')

        return transform_fns
                
    def __getitem__(self, idx):
        file_path = self.files[idx][0]
        MOSlabels = self.files[idx][1:] # geometry, texture, material, plausibility, artifact, preference
        data = np.load(file_path, allow_pickle=True).item()
        
        data_dict={}
        data_dict['coord'] = data['xyz']
        data_dict['color'] = data['rgb']
        data_dict['normal'] = data['normal']
        data_dict['roughness'] = data['roughness']
        data_dict['metallic'] = data['metallic']
        data_dict['mos'] = torch.from_numpy(np.array(MOSlabels[self.criterion]))

        # Coordinate normalize
        data_dict = pc_normalize(data_dict)
        
        # Augmentation
        for augment_fn in self.augment_fns:
            data_dict = augment_fn(data_dict)

        # Voxelize
        data_dict = self.gridsampler(data_dict)

        data_dict = to_tensor(shuffle_point(data_dict))
        
        data_dict = self.collect_keys(data_dict)
        
        
        return data_dict

def collate_fn(items):

    return items