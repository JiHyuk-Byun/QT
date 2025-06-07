from os import path as osp
from typing import Dict
import importlib
import io, time

import torch
from torch.utils.data import Dataset
import numpy as np

from .base import QA3DBaseDataModule
from .utils import read_csv, pc_normalize, mos_normalize, feat_normalize, to_tensor, shuffle_point, GridSample, Collect

MAX_RETRY = 3  
RETRY_SLEEP = 0.1 

class GC3DDataModule(QA3DBaseDataModule):

    def __init__(self, 
                 root_dir: str,
                 train_split: str,
                 test_split: str,
                 criterion: list,

                 batch_size: int,
                 num_workers: int,
                 dataset_config: dict,
                 eval_batch_size: int = -1,
                 name: str = 'gc3d'
                 ):
        super().__init__(name,
                         root_dir,
                         train_split,
                         test_split,
                         batch_size,
                         num_workers,
                         eval_batch_size)

        self.criterion = criterion
        self.dataset_config = dataset_config
    
    def _get_dataset(self, is_train: bool = True):
        split = 'train' if is_train else 'test'
        return GC3DDataset(**self.dataset_config, root_dir=self.root_dir, train_split=self.train_split, test_split=self.test_split, 
                           criterion=self.criterion, split=split)
    
class GC3DDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 train_split: str,
                 test_split: str,
                 split: str,
                 augments: Dict[str, dict],

                 criterion: list,
                 grid_size: int = 0.02,
                 hash_type: str = 'fnv',
                 return_grid_coord: bool = True,
                 
                 keys=['coord', 'grid_coord', 'mos', 'id'],
                 feat_keys=['coord', 'normal'],
                 manual_seed: int = None):
        
        super().__init__()
        self.root_dir = root_dir

        assert split in ['train', 'test']
        self.split = split
        self.train_split = osp.join(root_dir, 'train_split.csv') if train_split is None else train_split
        self.test_split = osp.join(root_dir, 'test_split.csv') if test_split is None else test_split
        
        self.split_path = self.train_split if self.split == 'train' else self.test_split
        
        self.manual_seed = manual_seed

        self.files = read_csv(self.split_path)
        self.criterion = criterion

        #Augmentation
        self.augment_fns = self._compose(augments)

        self.grid_sampler = GridSample(grid_size=grid_size, hash_type=hash_type, return_grid_coord=return_grid_coord, mode='train')
        
        self.feat_keys = feat_keys
        self.collect_keys = Collect(keys=keys, feat_keys=feat_keys)
        # after collect, e.g. key: [coord, grid_coord, 'mos', 'offset', 'feat'(which is concatnated)] 
    
    def _compose(self, augments: List[dict]):
        
        transform_fns = []
        if augments == None:
            return transform_fns

        utils = importlib.import_module('qt.data.utils')

        if isinstance(augments, list):
            for cfg in augments:
                cfg_local = cfg.copy()
                fn_name = cfg_local.pop('name')
                if hasattr(utils, fn_name):
                    transform_fns.append(getattr(utils, fn_name)(**cfg_local))
                else:
                    raise AttributeError(f'No function named {fn_name} defined in utils.py')

        elif isinstance(augments, dict):
            for fn_name, args in augments.items():
                if hasattr(utils, fn_name):
                    transform_fns.append(getattr(utils, fn_name)(**args))
                else:
                    raise AttributeError(f'No function named {fn_name} defined in utils.py')

        else:
            raise TypeError("augments must be list or dict")

        return transform_fns
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        '''
        coord: almost -1~1
        grid_coord: 0~98
        color: 0~1
        normal: -1~1
        metallic&roughness: 0~1 (No need to normalize)
        '''
        file_path = osp.join(self.root_dir, self.files[idx][0], 'features.npy')
        MOSlabels = self.files[idx][1]

        # Safe load
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

        data_dict = dict()
        data_dict['id'] = self.files[idx][0]
        
        # Load features.
        for feat in self.feat_keys:
            assert feat in ['coord', 'color', 'normal','metallic', 'roughness'], f'Invalid feature is in feat_keys, feature type: {feat}'
            if feat in ['metallic', 'roughness']:
                data_dict[feat] = data[feat][:, None].astype(np.float32)
            else:
                data_dict[feat] = data[feat].astype(np.float32)
        data_dict['mos'] = torch.tensor([MOSlabels]*len(self.criterion), # to make align with pre-trained sub-criteria model.
                                   dtype=torch.float32)
        
        data_dict = pc_normalize(data_dict)
        data_dict = feat_normalize(data_dict)
        #data_dict = mos_normalize(data_dict)
        
        #Augmentation
        for fn in self.augment_fns:
            data_dict = fn(data_dict)

        # Grid sampling
        data_dict = self.grid_sampler(data_dict)
        # Shuffle the order of pc and convert it to Tensor
        data_dict = to_tensor(shuffle_point(data_dict))
        # for feat in self.feat_keys:
        #     print('-'*40)
        #     print(f'key: {feat}')
        #     print(f'value: {data_dict[feat]}')

        #     print(f'shape: {data_dict[feat].shape}')
        #     print(f'scale: {data_dict[feat].min()}~{data_dict[feat].max()}')
            
        return self.collect_keys(data_dict)