from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):

    def __init__(self, root_dir: str, criterion: int, split: str, manual_seed: int):

        self.root_dir = root_dir
        self.criterion = criterion
        self.split = split
        
        self.randg = np.random.RandomState()

        if manual_seed is not None:
            self._reset_seed(manual_seed)

    def _reset_seed(self, seed):
        self.randg.seed(seed)

    