import os
import json

import MinkowskiEngine as ME
import xlrd
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def read_xlrd(excelFile):
  data = xlrd.open_workbook(excelFile)
  table = data.sheet_by_index(0)
  dataFile = []
  for rowNum in range(table.nrows):
    if rowNum > 0:
      dataFile.append(table.row_values(rowNum))
  return dataFile



class ObjaverseConfig(Dataset):
    def __init__(self, 
                root = '/media/ssd1/users/jhbyun/dataset/objaverse_pcs_sampled2', 
                npoints=2500, 
                voxel_size=5,
                split='train',
                transform=None,
                random_rotation=True,
                random_scale=True,
                manual_seed=False,
                ):
        
        
        self.phase = phase
        self.voxel_size = voxel_size
        self.transform = transform

        self.rotation_range = 
        self.random_scale = random_scale
        self.random_rotation = random_rotation
        self.randg = np.random.RandomState()

        if manual_seed:
            self.reset_seed()
    
    def reset_seed(self, seed=0):
        self.randg.seed(seed)
    
    def __len__(self):
        return len(self.files)

class ObjaverseResscnnDataset(ObjaverseConfig):
    def __init__(self,
                root = '/media/ssd1/users/jhbyun/dataset/objaverse_pcs_sampled2', 
                transform=None,
                split='train',
                voxel_size=5,
                random_rotation=True,
                random_scale=True,
                rotation_range=360,
                min_scale=0.8,
                max_scale=1.2,
                manual_seed=False):

    ObjaverseConfig.__init__(self, root = '/media/ssd1/users/jhbyun/dataset/objaverse_pcs_sampled2', 
                            voxel_size=voxel_size, split=split, transform=transform, 
                            random_rotation=random_rotation, random_scale=random_scale, manual_seed=manual_seed,)

    self.files = read_xlrd(osp.join(root, f'{split}_split.xlsx'))
    self.rotation_range = rotation_range

    def __getitem__(self, idx):
        file_path = self.files[idx][0]
        data = np.load(file_path)

        xyz = data['xyz']
        normal = data['normal'] # [-1, 1]
        rgb = data['rgb'] # [0, 1]
        roughness = data['roughness'] #[0, 255]
        metallic = data['metallic'] #[0, 255]

        if self.random_rotation:
            T0 = self._sample_random_trans(xyz, self.randg, self.rotation_range)
            xyz = self.apply_transform(xyz, T0)
        
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            xyz = scale * xyz

        # extract features
        feats = []
        # rgb
        feats.append(rgb - 0.5) 
        feats = np.hstack(feats) # Concat features

        # Extract occupied voxels
        xyz = torch.from_numpy(xyz)
        _, sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
        
        # Select coords and features
        coords = np.floor(xyz / self.voxel_size)
        coords = coords[sel]
        coords = ME.utils.batched_coordinates([coords])
        feats = feats[sel]

        feats = torch.as_tensor(feats, dtype=torch.float32)
        coords = torch.as_tensor(coords, dtype=torch.int32)
        
        if self.transform:
            coords, feats = self.transform(coords, feats)

        MOSlabels = self.files[idx][1:] # geometry, texture, material, plausibility, artifact, preference
        MOSlabel = torch.from_numpy(np.array(MOSlabel[5]))

        return (feats, coords, MOSlabel)

    # Rotation matrix along axis with angle theta
    def _M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    # Rigid Transformation
    def _sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

    def _apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts


if __name__ == '__main__':
    
    use_random_scale = False
    use_random_rotation = False
    transforms = []

    