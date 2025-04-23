import os
from os import path as osp

from PIL import Image as Image
import glob
from tqdm import tqdm
import trimesh
import numpy as np

SRC_PATH = '/workspace/dataset/objaverse_exp'
OUT_PATH = '/workspace/dataset/objaverse_pcs'

def _has_attr(obj, attr_name):
    return hasattr(obj, attr_name) and getattr(obj, attr_name) is not None

def _save_lst_as_txt(lst, filename):
    with open(filename, 'w') as f:
        f.write("\n".join(lst))

def _extract_pc_features(glb_path):

    has_uvs = True
    no_texture = False
    no_material = False
    scene_or_mesh = trimesh.load(glb_path)
    
    # In the case that mesh has multi-meshes
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    pos, face_indices = trimesh.sample.sample_surface(mesh, 150000)
    exit()
    ## === Extract Position ===
    positions = mesh.vertices

    ## === Extract Normal ===
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()
    normals = mesh.vertex_normals
    
    # Extract UV map for further feature extraction
    if hasattr(mesh.visual, "uv"):
        uvs = np.clip(mesh.visual.uv, 1e-6, 1 - 1e-6)
    else:
        has_uvs = False

    #print(dir(mesh.visual.material))
    ## === Extract Albedo === ##
    base_factor = mesh.visual.material.baseColorFactor if _has_attr(mesh.visual.material, "baseColorFactor") else [255, 255, 255, 255]
    base_factor = np.array(base_factor[:3], dtype=np.float32) / 255.0

    # Try vertex_colors
    if _has_attr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(positions):
        vertex_colors = mesh.visual.vertex_colors[:, :3]
    # Fallback: sample from texture using UV
    elif has_uvs and _has_attr(mesh.visual.material, "baseColorTexture"):
        tex = np.array(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float32) / 255.0 # texture image (PIL.Image)
        tex = tex * base_factor
        tex_h, tex_w = tex.shape[:2]
        
        px = (uvs[:, 0] * (tex_w - 1)).astype(np.int32)
        py = ((1 - uvs[:, 1]) * (tex_h - 1)).astype(np.int32)  # flip y-axis
        vertex_colors = tex[py, px]
    else:
        no_texture = True
        vertex_colors = np.full_like(positions, 255)  # white

    ## === Extract Material map features === ##
    r_factor = mesh.visual.material.roughnessFactor if _has_attr(mesh.visual.material, "roughnessFactor") else 1
    m_factor = mesh.visual.material.metallicFactor if _has_attr(mesh.visual.material, "metallicFactor") else 1

    if has_uvs and _has_attr(mesh.visual.material, "metallicRoughnessTexture"):
        mr_tex = np.array(mesh.visual.material.metallicRoughnessTexture.convert("RGB")) # texture image (PIL.Image)
        mr_h, mr_w = mr_tex.shape[:2]
        
        # map uv value into real number coordinates.
        px = (uvs[:, 0] * (mr_w - 1)).astype(np.int32)
        py = ((1 - uvs[:, 1]) * (mr_h - 1)).astype(np.int32)  # flip y-axis
        
        vertex_mrs = mr_tex[py, px]
        roughness = vertex_mrs[:, 1] * r_factor
        metallic = vertex_mrs[:, 2] * m_factor

    else:
        # Default mr map
        no_material = True
        roughness = np.full((len(positions),), 1.0) * r_factor  # rough surface
        metallic = np.full((len(positions),), 1.0) * m_factor # non-metallic
    
    return positions, normals, vertex_colors, roughness, metallic, no_texture, no_material

def main():

    data_dict = {}

    # read data
    for subfolder in os.listdir(SRC_PATH):
        subfolder_path = osp.join(SRC_PATH,subfolder)
        if osp.isdir(subfolder_path):
            file_paths = [
                osp.join(subfolder_path, f)
                for f in os.listdir(subfolder_path)
                if osp.isfile(osp.join(subfolder_path, f))
            ]
            data_dict[subfolder] = file_paths
    
    no_texture_lst = []
    no_material_lst = []

    for key, glb_lst in tqdm(data_dict.items()):
        print('current directory: ', key)
        for glb in tqdm(glb_lst):
            glb = glb.split('/')[-1]
            glb_file = osp.join(key, glb)
            glb_id = glb_file.split('.')[0]
            glb_path = osp.join(SRC_PATH, glb_file)

            out_path = osp.join(OUT_PATH, glb_id)
            os.makedirs(osp.dirname(out_path), exist_ok=True)

            xyz, normal, rgb, roughness, metallic, \
                flag_no_texture, flag_no_material = _extract_pc_features(glb_path)

            if flag_no_texture:
                no_texture_lst.append(glb_id)
            if flag_no_material:
                no_material_lst.append(glb_id)
            
            save_data = {'xyz': xyz,
                         'normal': normal,
                         'rgb': rgb,
                         'roughness': roughness,
                         'metallic': metallic}
            np.save(out_path, save_data, allow_pickle=True)

    _save_lst_as_txt(no_texture_lst, osp.join(OUT_PATH, "no_textures.txt"))
    _save_lst_as_txt(no_material_lst, osp.join(OUT_PATH, "no_materials.txt"))

if __name__ == '__main__':
    main()
    print('Done!')
# 0b2bd5fe961e42b584648101e51bd68f: metal0, rough1인 전형적인 애들
# 00b2c8c60d2f45a893ee73fd1f107e27: mr이 아니라 specular로 저장되어 있음. -> 근데 attribute자체는 동일하게 되어있고 따로 specular라는 이름을 가지지는 않음.
# 0b20384a14ac4c4c9dc083013b7c1598: 이거는 mr있는데? metallicRoughnessTexture이 None이더라도, black으로 채우고 metallicRoughnessFactor를 각각 곱한결과를 최종 map으로 가져오는듯

