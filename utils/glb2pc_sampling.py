import os
from os import path as osp
import traceback

import open3d as o3d
from PIL import Image as Image
import glob
from tqdm import tqdm
import trimesh
import numpy as np

from multiprocessor import MultiProcessRunner

SRC_PATH = '/mnt/volume4/users/join/objaverse/hf-objaverse-v1/glbs'
OUT_PATH = '/mnt/volume4/users/join/objaverse/hf-objaverse-v1/pcs2' #_1250_fps'
SAVE_PLY = False
SAVE_ITEM = True
N_SAMPLE = 60000

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

def _save_lst_as_txt(lst, filename):
    with open(filename, 'w') as f:
        f.write("\n".join(lst))

def _has_attr(obj, attr_name):
    return hasattr(obj, attr_name) and getattr(obj, attr_name) is not None

# def _furthest_point_sampling(points: np.ndarray, k: int):

#     n = points.shape[0]
#     centroids = np.zeros(k, dtype=np.int64)
#     distance  = np.full(n, 1e10, dtype=points.dtype)

#     # 시작점 무작위(한 점 지정)
#     farthest = np.random.randint(0, n)

#     for i in range(k):
#         centroids[i] = farthest                    # 현재 farthest 저장
#         centroid = points[farthest][None, :]       # (1, 3)
#         dist = np.sum((points - centroid) ** 2, 1) # 모든 점과의 거리
#         mask = dist < distance                     # 더 가까우면 갱신
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance)             # 가장 먼 점 선택

#     return points[centroids], centroids

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

def sample_pc_features(glb_path, n_points):

    has_uvs = True
    no_texture = False
    no_material = False
    no_attr = False
    scene_or_mesh = trimesh.load(glb_path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh
    ## === Sample Position ===
    positions, face_indices = trimesh.sample.sample_surface_even(mesh, n_points)
    
    
    faces = mesh.faces[face_indices]
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
###    
    # raw_pts, face_idx = mesh.sample(n_points, return_index=True)

    # positions, fps_indices = _furthest_point_sampling(raw_pts, n_points)

    # faces = mesh.faces[face_idx[fps_indices]]
    # v0, v1, v2 = mesh.vertices[faces[:, 0]], mesh.vertices[faces[:, 1]], mesh.vertices[faces[:, 2]]

    ## === Extract Normal ===
    normals = mesh.face_normals[face_indices]
    
    # UV map
    if _has_attr(mesh.visual, "uv"):
        uv0 = mesh.visual.uv[faces[:, 0]]
        uv1 = mesh.visual.uv[faces[:, 1]]
        uv2 = mesh.visual.uv[faces[:, 2]]
        
        bary_weights = _barycentric_weights(positions, v0, v1, v2)
        uvs = (bary_weights[:, [0]] * uv0 +
              bary_weights[:, [1]] * uv1 +
              bary_weights[:, [2]] * uv2
              )
        uvs = np.clip(uvs, 1e-6, 1 - 1e-6)
    else:
        has_uvs = False
    
    #print(dir(mesh.visual.material))

    if _has_attr(mesh.visual, "material"):
        ## === Extract Albedo === ##
        base_factor = mesh.visual.material.baseColorFactor if _has_attr(mesh.visual.material, "baseColorFactor") else [255, 255, 255, 255]
        base_factor = np.array(base_factor[:3], dtype=np.float32) / 255.0

        if has_uvs and _has_attr(mesh.visual.material, "baseColorTexture"):
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
        r_factor = mesh.visual.material.roughnessFactor if not no_attr and _has_attr(mesh.visual.material, "roughnessFactor") else 1
        m_factor = mesh.visual.material.metallicFactor if not no_attr and _has_attr(mesh.visual.material, "metallicFactor") else 1

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
            roughness = np.full((len(positions),), 1.0)  # rough surface
            metallic = np.full((len(positions),), 1.0) # non-metallic
    else:
        no_attr = True
        vertex_colors = np.full_like(positions, 255)
        roughness = np.full((len(positions),), 1.0)  # rough surface
        metallic = np.full((len(positions),), 1.0) # non-metallic

    
    
    return positions, normals, vertex_colors, roughness, metallic, no_texture, no_material, no_attr

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
    
    error_lst = []
    no_texture_lst = []
    no_material_lst = []
    no_attr_lst = []
    log_dir = osp.join(OUT_PATH,"error_logs")
    os.makedirs(log_dir,exist_ok=True)

    if SAVE_ITEM:
        _save_lst_as_txt([osp.join(osp.basename(k), osp.splitext(v)[0]) for k, values in data_dict.items() for v in values], osp.join(OUT_PATH, 'items.txt'))

    for key, glb_lst in tqdm(data_dict.items()):
        print('current directory: ', key)
        for glb in tqdm(glb_lst):
            glb = glb.split('/')[-1]
            glb_file = osp.join(key, glb)
            glb_id = glb_file.split('.')[0]
            glb_path = osp.join(SRC_PATH, glb_file)

            out_path = osp.join(OUT_PATH, glb_id)
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            if osp.exists(out_path + '.npy'):
                continue
            
            try:
                xyz, normal, rgb, roughness, metallic, \
                    flag_no_texture, flag_no_material, flag_no_attr = sample_pc_features(glb_path, N_SAMPLE)
                
                if flag_no_texture:
                    no_texture_lst.append(glb_id)
                if flag_no_material:
                    no_material_lst.append(glb_id)
                if flag_no_attr:
                    no_attr_lst.append(glb_id)

                save_data = {'coord': xyz.astype(np.float32),
                            'normal': normal.astype(np.float32),
                            'color': rgb.astype(np.float32),
                            'roughness': roughness.astype(np.float32),
                            'metallic': metallic.astype(np.float32)}
                
                np.save(out_path, save_data, allow_pickle=True)
                if SAVE_PLY:
                    save_pointcloud_as_ply(save_data['coord'], save_data['color'], save_data['normal'],filename=out_path+'.ply')

            except Exception as e: 
                log_filename = f"{glb_id}.txt"
                error_lst.append(glb_id)
                log_path = osp.join(log_dir, log_filename)

                os.makedirs(osp.dirname(log_path), exist_ok=True)

                with open(log_path, 'w') as f:
                    f.write(f"! Error while processing: {glb_path}\n\n")
                    f.write(traceback.format_exc())
                
            # print(f"glb: {glb}")
            # print(f"xyz: {save_data['xyz'].shape}")
            # print(f"normal: {save_data['normal'].shape}")
            # print(f"rgb: {save_data['rgb'].shape}")
            # print(f"roughness: ", save_data['roughness'].shape) #{any(x != 0 for x in save_data['roughness'])}")
            # print(f"metallic: ", save_data['metallic'].shape) #{any(x != 0 for x in save_data['metallic'])}")

    _save_lst_as_txt(no_texture_lst, osp.join(OUT_PATH, "no_textures.txt"))
    _save_lst_as_txt(no_material_lst, osp.join(OUT_PATH, "no_materials.txt"))
    _save_lst_as_txt(no_attr_lst, osp.join(OUT_PATH, "no_attrs.txt"))
    _save_lst_as_txt(error_lst, osp.join(OUT_PATH, "error_items.txt"))

if __name__ == '__main__':
    main()
    print('Done!')

# 여러 mesh로 이루어진 scene에 대해서 texture flag처리가 잘 안되는거 같은데