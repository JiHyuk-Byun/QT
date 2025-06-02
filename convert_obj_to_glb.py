import glob
from os import path

import trimesh
from tqdm import tqdm

ROOT_DIR = 'datasets/3DGCQA_master/obj'


def main():
    files = glob.glob(path.join(ROOT_DIR, '**/**/', '*.obj'), recursive=True)
    for file in tqdm(files):
        mesh = trimesh.load(file)
        save_path = file.replace('.obj', '.glb')
        mesh.export(save_path, file_type='glb')


if __name__ == '__main__':
    main()
