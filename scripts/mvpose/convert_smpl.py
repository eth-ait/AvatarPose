
import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
from AvatarPose.utils.file_utils import read_json, write_keypoints3d
from glob import glob
from instant_avatar.deformers.smplx.body_models import SMPL

def body25_to_skel19(joints):
    map_idx = np.array([8, 1, 9, 12, 0, 2, 5, 10, 13, 17, 18, 3, 6, 11, 14, 4, 7, 19, 22], dtype=np.int32)
    return joints[..., map_idx, :]

# from matching import match_3d_greedy, get_matching_dict
start_frame = 3
regressor = np.load('data/SMPLX/J_regressor_body25.npy')

model = SMPL(gender='neutral', model_path='data/SMPLX/smpl').cuda()
faces = model.faces

def save_keypoints(name, database, smplname, kptsname):
    print(name)
    input_data = f"{database}/{name}/{smplname}"
    output_folder =  f"{database}/{name}/{kptsname}"
    os.makedirs(output_folder, exist_ok=True)
    smplnames = sorted(glob(os.path.join(input_data, '*.json')))
    frames = [int(os.path.basename(smplname).split('.')[0]) for smplname in smplnames]
    idxs = np.argsort(frames)
    smplnames = [smplnames[idx] for idx in idxs]
    for frame_idx in tqdm(range(len(smplnames))):
        smplname = smplnames[frame_idx]
        frame = frames[frame_idx]
        data = read_json(smplname)
        keypoints_all = []
        for smpl in data:
            keypoints_dict = {}
            keypoints_dict['id'] = smpl.pop('id')
            smpl = {key: np.array(smpl[key]).reshape(1, -1) for key in smpl.keys()}
            smpl_torch = {key: torch.from_numpy(smpl[key]).to(torch.float32).cuda() for key in smpl.keys()}
            output = model(betas=smpl_torch['betas'], 
                             body_pose=smpl_torch['body_pose'],
                             global_orient=smpl_torch['global_orient'],
                             transl=smpl_torch['transl'])
            vertices = output.vertices[0].detach().cpu().numpy()
            joints = np.matmul(regressor, vertices)
            skel19 = body25_to_skel19(joints)
            keypoints_dict['keypoints3d'] = skel19
            keypoints_all.append(keypoints_dict)
        write_keypoints3d(os.path.join(output_folder, f'{frame:06d}.json'), keypoints_all)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    save_keypoints(args.seq, args.data_root, 'smpl_init', 'skel19_fit')
    
    