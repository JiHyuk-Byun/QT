import os
from os import path as osp
from tqdm import tqdm

SRC_PATH = '/mnt/volume4/users/join/objaverse/hf-objaverse-v1/glbs'
ITEMS_TXT = '/workspace/dataset/objaverse_userstudy/items.txt'
TGT_PATH = '/workspace/dataset/objaverse_userstudy'
#N_FILES = 12500

def main():

    with open(ITEMS_TXT, 'r') as f:
        items = f.read().strip().split('\n')

    items = sorted(items)
    item_paths = list(map(lambda x: osp.join(SRC_PATH, x), items[:]))

    for path in tqdm(item_paths):

        dir_id = path.split('/')[-2]
        out_dir = osp.join(TGT_PATH, dir_id)
        os.makedirs(out_dir, exist_ok=True)

        cmd = f"cp {path} {out_dir}"
        os.system(cmd)

    print('Done!')

main()