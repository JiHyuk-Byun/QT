import os
from os import path as osp
import traceback
import csv

import open3d as o3d
from PIL import Image as Image
import glob
from tqdm import tqdm
import trimesh
import numpy as np

from multiprocessor import MultiProcessRunner

SRC_PATH = '/workspace/dataset/3DGCQA_master/obj'
CSV_PATH = '/workspace/dataset/3DGCQA_master/3DGCQA-mos.csv'
OUT_PATH = '/workspace/dataset/3DGC_dataset' 
SAVE_PLY = True
SAVE_ITEM = True
N_SAMPLE = 60000

def _norm_to_rgb(normals):
    """[-1,1] 범위 → [0,1] 컬러"""
    return (normals * 0.5 + 0.5).clip(0, 1)

def _single_to_gray(vals):
    """0~1 스칼라 → 회색"""
    return np.repeat(vals[:, None], 3, axis=1)

def _single_to_cyan(vals):
    """0~1 스칼라 → 금속(시안)"""
    rgb = np.zeros((len(vals), 3), dtype=np.float32)
    rgb[:, 1:] = vals[:, None]        # G, B 채널만
    return rgb

def save_pointcloud_as_ply(positions, colors=None, normals=None, filename="output.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    if colors is not None:
        # normalize if needed
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(filename, pcd)
    #print(f"Saved: {filename}")

def _read_csv(csv_path):
    items = dict()
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0: continue
            items[row[1]] = row[2]
    
    return items

def _save_lst_as_txt(lst, filename):
    with open(filename, 'w') as f:
        f.write("\n".join(lst))

def _has_attr(obj, attr_name):
    return hasattr(obj, attr_name) and getattr(obj, attr_name) is not None

# Any points belongs to face can be represented by the ar
def _barycentric_weights(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    # dot product
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d02 = np.sum(v0 * v2, axis=1)
    d12 = np.sum(v1 * v2, axis=1)

    area_tot = d00 * d11 - d01 * d01
    area_pca = d11 * d02 - d01 * d12
    area_pba = d00 * d12 - d01 * d02

    v = area_pca / area_tot
    w = area_pba / area_tot
    u  = 1 - w - v

    return np.stack([u, v, w], axis=1)

def _parse_mtl(obj_path):
    '''
    returns path of texture map with Dict
    '''
    tex = {}
    obj_dir = osp.dirname(obj_path)
    mtl_file = None
    with open(obj_path, 'r') as f:
        for line in f:
            if line.lower().startswith('mtllib'):
                mtl_file = line.split(maxsplit=1)[1].strip()
                break
    if mtl_file is None:
        return tex
    
    mtl_path = osp.join(obj_dir, mtl_file)

    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            t = line.strip().split()
            if not t:
                continue
            key = t[0].lower()

            if key in {"map_kd"} and 'albedo' not in tex:
                tex['albedo'] = osp.join(obj_dir, " ".join(t[1:]))
            elif key in {'map_bump', 'bump'} and 'normal' not in tex:
                tex['normal'] = osp.join(obj_dir, " ".join(t[1:]))
            elif key in {"map_pr"} and 'roughness' not in tex:
                tex['roughness'] = osp.join(obj_dir, " ".join(t[1:]))
            elif key in {"map_pm"} and 'metallic' not in tex:
                tex['metallic'] = osp.join(obj_dir, " ".join(t[1:]))

    return tex


def _sample_tex(tex_im, uvs):
    """
    • tex_im : H×W×3 numpy float32 (0–1)
    • uvs    : N×2 numpy (0–1)   (OpenGL 기준, V 위아래 반전 필요)
    """
    if tex_im is None:
        return None
    h, w = tex_im.shape[:2]
    px = np.clip((uvs[:, 0] * (w - 1)).round().astype(np.int32), 0, w - 1)
    py = np.clip(((1 - uvs[:, 1]) * (h - 1)).round().astype(np.int32), 0, h - 1)
    return tex_im[py, px]

def sample_obj_features(item_path, n_points):

    obj_path = osp.join(item_path, 'model.obj')
    flags = {"no_uv": False, "no_albedo": False,
             "no_normal": False, "no_mr": False}

    obj = trimesh.load(obj_path, force='mesh')

    if isinstance(obj, trimesh.Scene):
        mesh = obj.dump(concatenate=True)
    else:
        mesh = obj
    ## === Sample Position ===
    positions, face_indices = trimesh.sample.sample_surface_even(mesh, n_points)

    faces = mesh.faces[face_indices]
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]


    # UV map
    if not _has_attr(mesh.visual, "uv"):
        flags['no_uv'] = True
        uvs = None
    else:
        uv0 = mesh.visual.uv[faces[:, 0]]
        uv1 = mesh.visual.uv[faces[:, 1]]
        uv2 = mesh.visual.uv[faces[:, 2]]
        
        bary_weights = _barycentric_weights(positions, v0, v1, v2)
        uvs = (bary_weights[:, [0]] * uv0 +
              bary_weights[:, [1]] * uv1 +
              bary_weights[:, [2]] * uv2
              )
        uvs = np.clip(uvs, 1e-6, 1 - 1e-6)
    
    #print(dir(mesh.visual.material))
    # Load Texture maps
    tex_paths = _parse_mtl(obj_path)

    load = lambda p: np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0 if p and osp.exists(p) else None
    tex_albedo   = load(tex_paths.get("albedo"))
    tex_normal   = load(tex_paths.get("normal"))
    tex_rough    = load(tex_paths.get("roughness"))
    tex_metallic = load(tex_paths.get("metallic"))

    if uvs is not None and tex_albedo is not None:
        colors = _sample_tex(tex_albedo, uvs)
    elif _has_attr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[faces[:, 0], :3] /255.0
    else:
        flags['no_albedo'] = True
        colors = np.ones_like(positions, dtype=np.float32)

    if uvs is not None and tex_normal is not None:
        nmap = _sample_tex(tex_normal, uvs) * 2.0 - 1.0
        normals = nmap
    elif _has_attr(mesh, 'face_normals'): 
        normals = normals = mesh.face_normals[face_indices]
    else:
        flags['no_normal'] = True

    if uvs is not None and (tex_rough is not None or tex_metallic is not None):
        r = _sample_tex(tex_rough, uvs)[:, 0] 
        m = _sample_tex(tex_metallic, uvs)[:, 0]
        roughness, metallic = r, m
    else:
        flags['no_mr'] = True
        roughness = np.ones(n_points, dtype=np.float32)
        metallic  = np.ones(n_points, dtype=np.float32)
        
    
    return  {'coord': positions.astype(np.float32),
            'normal': normals.astype(np.float32),
            'color': colors.astype(np.float32),
            'roughness': roughness.astype(np.float32),
            'metallic': metallic.astype(np.float32)}, flags

def main():

    data_dict = {}

    item_dict= _read_csv(CSV_PATH)

    
    error_lst = []
    
    no_albedo_lst = []
    no_mr_lst = []
    no_normal_lst = []
    no_uv_lst = []
    log_dir = osp.join(OUT_PATH,"error_logs")
    os.makedirs(log_dir,exist_ok=True)
    for item, mos in tqdm(item_dict.items()):
        item = item.replace('\\', '/')
        item_path = osp.join(SRC_PATH, item)
        out_path = osp.join(OUT_PATH, item, 'features')
        #print(out_path)
        #exit()
        if osp.exists(out_path+'.npy'):
            continue
        try:

            features, flags = sample_obj_features(item_path, N_SAMPLE)
            
            if flags['no_uv']:
                no_uv_lst.append(item)
            if flags['no_albedo']:
                no_albedo_lst.append(item)
            if flags['no_normal']:
                no_normal_lst.append(item)
            if flags['no_mr']:
                no_mr_lst.append(item)
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            np.save(out_path, features, allow_pickle=True)
            if SAVE_PLY:
                base = out_path                      # …/features
                # ① 알베도
                save_pointcloud_as_ply(features['coord'],
                                    colors=features['color'],
                                    normals=features['normal'],
                                    filename=base + '_albedo.ply')
                # ② 노멀
                save_pointcloud_as_ply(features['coord'],
                                    colors=_norm_to_rgb(features['normal']),
                                    filename=base + '_normalRGB.ply')
                # ③ 러프니스
                save_pointcloud_as_ply(features['coord'],
                                    colors=_single_to_gray(features['roughness']),
                                    filename=base + '_roughness.ply')
                # ④ 메탈릭
                save_pointcloud_as_ply(features['coord'],
                                    colors=_single_to_cyan(features['metallic']),
                                    filename=base + '_metallic.ply')
        except Exception as e: 
            print('error!')
            log_filename = f"{item}.txt"
            error_lst.append(item)
            log_path = osp.join(log_dir, log_filename)

            os.makedirs(osp.dirname(log_path), exist_ok=True)

            with open(log_path, 'w') as f:
                f.write(f"! Error while processing: {item_path}\n\n")
                f.write(traceback.format_exc())

if __name__ == '__main__':
    main()
    print('Done!')

# 여러 mesh로 이루어진 scene에 대해서 texture flag처리가 잘 안되는거 같은데