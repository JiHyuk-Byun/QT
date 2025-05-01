import json

JSON_PATH = '/workspace/dataset/objaverse_userstudy/merged_scores_filtered.json'
GOBJ2OBJ_PATH = '/workspace/dataset/gobjaverse_280k_index_to_objaverse.json'
CRITERIA = ['geometry', 'texture', 'material', 'plausibility', 'artifacts', 'preference']
OUT_JSON = '/workspace/dataset/objaverse_userstudy/parsed_userstudy.json'

def main():

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    with open(GOBJ2OBJ_PATH, 'r') as f:
        gobj2obj = json.load(f)
    

    parsed_data = {}

    for metadata in data:
        key = gobj2obj[metadata['gobjaverse_index']]
        score_dict = {criterion: metadata[criterion]['mos'] for criterion in CRITERIA}
        parsed_data[key] = score_dict
    
    
    with open(OUT_JSON, 'w') as f:
        json.dump(parsed_data, f, indent=4)
    
    print('Done!')

main()
