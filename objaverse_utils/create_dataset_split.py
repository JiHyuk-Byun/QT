import os
from os import path as osp
import random

import json
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

SRC_DIR = '/workspace/dataset/objaverse_pcs_12500'
OUT_DIR = '/workspace/dataset/objaverse_pcs_12500'
LABEL_PATH = '/workspace/dataset/vlm_results/parsed_index_v2.json'
CRITERIA = ['geometry', 'texture', 'material', 'plausibility', 'artifacts', 'preference']
TEST_RATIO = 0.2

def extract_id(path):
    dict_id = path.split('/')[-2].split('-')[0] + path.split('/')[-2].split('-')[1]
    data_id = path.split('/')[-1].split('.')[0]
    return dict_id + data_id

def get_label(path, data, criterion):
    objaverse_index = osp.join(path.split('/')[-2], path.split('/')[-1].split('.')[0])
    try:
        score = data[objaverse_index+'.glb'][criterion]
    except Exception as e:
        
        print(f"{path}: {e}")
        exit()
    return score

def delete_no_attrs(data):
    with open(osp.join(SRC_DIR, 'no_attrs.txt'), 'r') as f:
        no_attrs_items = f.read().strip().split('\n')
    
    no_attrs_items = list(map(lambda x: osp.join(SRC_DIR, x+'.npy'), no_attrs_items))
    no_attrs_set = set(no_attrs_items)                  
    filtered_data = [x for x in data if x not in no_attrs_set]

    return filtered_data

def create_dataset_split(data_dir, output_dir=None, test_ratio=0.2, seed=42, save_csv=True):
    random.seed(seed)
    files = glob.glob(osp.join(data_dir, '*', '*.npy'))
    files = sorted(files, key=extract_id)

    filtered_files = delete_no_attrs(files)

    train_files, test_files = train_test_split(filtered_files, test_size=test_ratio, random_state=seed)

    print('loading data...')
    with open(LABEL_PATH, 'r') as f:
        data = json.load(f)

    print('extract labels...')
    train_labels = {
        criterion: [get_label(fp, data, criterion) for fp in train_files]
        for criterion in CRITERIA
    }
    test_labels = {
        criterion: [get_label(fp, data, criterion) for fp in test_files]
        for criterion in CRITERIA
    }

    if save_csv:
        df_train = pd.DataFrame({
            "filename": train_files,
            **train_labels
        })
        df_test = pd.DataFrame({
            "filename": test_files,
            **test_labels
        })

        df_train.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
        df_test.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    else:
        with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
            for fname, label in zip(train_files, train_labels):
                f.write(f"{fname} {label}\n")
        with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
            for fname, label in zip(test_files, test_labels):
                f.write(f"{fname} {label}\n")

    print(f"Train: {len(train_files)} samples")
    print(f"Test:  {len(test_files)} samples")

create_dataset_split(
    data_dir=SRC_DIR,
    output_dir=OUT_DIR,
    test_ratio=TEST_RATIO,
    save_csv=True
)