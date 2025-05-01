import os
from os import path as osp
import json

import pandas as pd
JSON_PATH = '/workspace/dataset/objaverse_userstudy/parsed_userstudy.json'
SRC_DIR = '/workspace/dataset/objaverse_pcs_userstudy'
OUT_DIR = '/workspace/dataset/objaverse_pcs_userstudy'

def main():

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    parsed_data = {'filename': []}

    for key, score_dict in data.items():
        npy_path = osp.splitext(key)[0] + '.npy'
        parsed_data['filename'].append(osp.join(SRC_DIR, npy_path))
        for criterion, score in score_dict.items():
            if criterion not in parsed_data:
                parsed_data[criterion] = []
            parsed_data[criterion].append(score)
    
    df = pd.DataFrame(parsed_data)

    df.to_excel(osp.join(OUT_DIR, 'test_split.xlsx'), index=False)

    print(f"Test: {len(parsed_data['filename'])} samples")

main()