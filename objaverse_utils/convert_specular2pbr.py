import os
from os import path as osp

import trimesh
import numpy as np

ITEMS_TXT = '/workspace/dataset/objaverse_pcs_12500/no_materials.txt'
SRC_PATH = '/workspace/dataset/objaverse_12500'
OUT_PATH = '/workspace/dataset/objaverse_pcs_12500'

def main():

    glb_id = '000-003/669d10273fc24bc4850dd7039be4e4ca.glb'
    glb_path = osp.join(SRC_PATH, glb_id)

    scene_or_mesh = trimesh.load(glb_path)
    print(scene_or_mesh)
    if isinstance(scene_or_mesh, trimesh.Scene):

        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh
    
    postions, face_indices = mesh.sample(150000, return_index=True)
    print(dir(mesh))
    # metallic, base_color, roughness = mesh.visual.gloss.specular_to_pbr()

    # print("m:", metallic)
    # print("b:", base_color)
    # print("r:", roughness)
main()