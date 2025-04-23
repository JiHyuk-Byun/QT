import os
from os import path as osp
import json


def main():
    with open('/workspace/dataset/vlm_results/qa_index_v2.json', 'r') as f:
        data = json.load(f)

    result = {}
    items = []
    for d in data:
        items.append(d['metadata']['objaverse_index'])
        result[d['metadata']['gobjaverse_index']] = d['metadata']['objaverse_index']

    with open('/workspace/dataset/gobj2obj.json', 'w') as f:
        json.dump(result, f, indent=4)

    with open('/workspace/dataset/indexv2_items.txt', 'w') as f:
        f.write('\n'.join(items))
    
    print(f"{len(result)} items are extracted!")

main()