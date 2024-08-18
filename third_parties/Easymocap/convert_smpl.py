
from ast import arg
from distutils.log import info
import os
import json
import torch
import argparse
import trimesh
import tqdm
import numpy as np
import shutil
from pathlib import Path
from easymocap.mytools.file_utils import read_json, write_keypoints3d
from easymocap.smplmodel.body_param import load_model
from glob import glob

def body25_to_skel19(joints):
    map_idx = np.array([8, 1, 9, 12, 0, 2, 5, 10, 13, 17, 18, 3, 6, 11, 14, 4, 7, 19, 22], dtype=np.int32)
    return joints[..., map_idx, :]

# from matching import match_3d_greedy, get_matching_dict
start_frame = 3
regressor = np.load('data/smplx/J_regressor_body25.npy')

model = load_model(gender='neutral', use_cuda=True, model_type='smpl', model_path='data/smplx')
faces = model.faces

def save_keypoints(name):
    input_data = f"./Hi4D_AvatarPose/{name}/smpl_fit"
    output_folder =  f"./Hi4D_AvatarPose/{name}/skel19_fit"
    # os.makedirs(output_folder / "0", exist_ok=True)
    # os.makedirs(output_folder / "1", exist_ok=True)
    # os.makedirs(output_folder / "params", exist_ok=True)

    smplnames = sorted(glob(os.path.join(input_data, '*.json')))
    frames = [int(os.path.basename(smplname).split('.')[0]) for smplname in smplnames]
    idxs = np.argsort(frames)
    smplnames = [smplnames[idx] for idx in idxs]
    for frame_idx in range(len(smplnames)):
        smplname = smplnames[frame_idx]
        frame = frames[frame_idx]
        data = read_json(smplname)
        keypoints_all = []
        for smpl in data:
            keypoints_dict = {}
            keypoints_dict['id'] = smpl['id']
            smpl = {key: np.array(smpl[key]) for key in smpl.keys()}
            smpl_torch = {key: torch.from_numpy(smpl[key]).to(torch.float32).cuda() for key in smpl.keys()}
            vertices = model(shapes=smpl_torch["shapes"],
                                poses=smpl_torch["poses"],
                                Rh=smpl_torch["Rh"],
                                Th=smpl_torch["Th"], 
                                return_verts=True, return_tensor=False)[0]
            joints = np.matmul(regressor, vertices)
            skel19 = body25_to_skel19(joints)
            keypoints_dict['keypoints3d'] = skel19
            keypoints_all.append(keypoints_dict)
        write_keypoints3d(os.path.join(output_folder, f'{frame:06d}.json'), keypoints_all)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    save_keypoints(args.name)