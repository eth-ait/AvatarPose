import numpy as np
import os
from instant_avatar.deformers.smplx import SMPL
import torch
from tqdm import tqdm
import hydra
from AvatarPose.utils.file_utils import write_keypoints3d, read_json
from os.path import join
from instant_avatar.deformers.smplx import SMPL
np.random.seed(2023)

REGRESSOR25 = np.load(hydra.utils.to_absolute_path('data/SMPLX/J_regressor_body25.npy')) # (25, 6890)
body25_skel19 = np.array([8, 1, 9, 12, 0, 2, 5, 10, 13, 17, 18, 3, 6, 11, 14, 4, 7, 19, 22], dtype=np.int32)

def get_keypoints3d(root):
    #{pid: keypoints3d, pid: keypoints3d}
    if not os.path.exists(root):
        return {}
    data = read_json(root)
    keypoints3d_dict = {'names_all':[]}
    for i in range(len(data)):
        pid = data[i]['id']
        keypoints3d = data[i]['keypoints3d']
        name = f'human_{pid}'
        keypoints3d_dict[name] = {}
        keypoints3d_dict['names_all'].append(name)
        keypoints3d = np.array(keypoints3d, dtype=np.float32)[None, :]
        keypoints3d_dict[name]['kpts'] = keypoints3d
    return keypoints3d_dict

def write_keypoints3d_txt(outname, keypoints3d_list_all):
    # keypoints3d_dict[key]: (n_frame, n_joints, 3)
    with open(outname, 'w') as f:
        n_joints = keypoints3d_list_all[0][0]['keypoints3d'].shape[0]
        n_frames = len(keypoints3d_list_all)
        f.write(str(n_joints) + '\t' + str(n_frames) + '\n')
        for frame_idx in range(n_frames):
            keypoints3d_list = keypoints3d_list_all[frame_idx]
            f.write(str(len(keypoints3d_list)) + '\n')
            for keypoints3d_dict in keypoints3d_list:
                f.write(str(keypoints3d_dict['id']) + '\n')
                keypoints3d = keypoints3d_dict['keypoints3d']
                for i in range(3):
                    for joint_idx in range(n_joints):
                        f.write(str(keypoints3d[joint_idx, i]) + '\t')
                    f.write('\n')
                # homogeneous
                for joint_idx in range(n_joints):
                    f.write(str(1) + '\t')
                f.write('\n')

def smpl_to_joint_model(model_type):
    if model_type == 'body25_joints':
        return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34], dtype=np.int32) # len = 25
    if model_type == 'shelf15':
        return np.array([8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 24, 0], dtype=np.int32) # len = 15
    if model_type == 'skel19':
        return np.array([0, 12, 2, 1, 24, 17, 16, 5, 4, 27, 28, 19, 18, 8, 7, 21, 20, 29, 32], dtype=np.int32)

def smplx_to_skel19():
    return np.array([0, 12, 2, 1, 55, 17, 16, 5, 4, 58, 59, 19, 18, 8, 7, 21, 20, 60, 63], dtype=np.int32)

def smpljoints_to_modeljoints(joints, model_type='body25_joints'):
    map_idx = smpl_to_joint_model(model_type)
    joints = joints.reshape(-1, 3)
    return joints[map_idx]

def body25_to_skel19(joints):
    return joints[..., body25_skel19, :]

def skel19_to_panoptic(joints):
    map_idx = np.array([1, 4, 0, 6, 12, 16, 3, 8, 14, 5, 11, 15, 2, 7, 13], dtype=np.int32)
    return joints[..., map_idx, :]

def skel19_to_shelf14(joints):
    map_idx = np.array([13, 7, 2, 3, 8, 14, 15, 11, 5, 6, 12, 16, 1, 4, 0], dtype=np.int32)
    shelf14 = joints[..., map_idx, :].reshape(15, 3)
    joints = joints.reshape(19, 3)
    faceDir_0 = np.cross(shelf14[12] - shelf14[14], shelf14[8] - shelf14[9])
    norm = np.linalg.norm(faceDir_0)
    faceDir = faceDir_0 / norm
    zDir = np.array([0, 0, 1], dtype=np.float32)
    shoulderCenter = (joints[5] + joints[6]) / 2.0
    headCenter = (joints[9] + joints[10])/2.0
    shelf14[12] = shoulderCenter + (headCenter - shoulderCenter) * 0.5
    shelf14[13] = shelf14[12] + faceDir * 0.125 + zDir * 0.145
    
    return shelf14

def body25_to_panoptic(joints):
    map_idx = np.array([1, 0, 8, 5, 6, 7, 12, 13, 14, 2, 3, 4, 9, 10, 11], dtype=np.int32)
    return joints[..., map_idx, :]

def smplvertices_to_body25(vertices):
    regressor25 = torch.from_numpy(REGRESSOR25).cuda()
    vertices = torch.from_numpy(vertices.reshape(-1, 3)).cuda()
    return (regressor25 @ vertices).detach().cpu().numpy() # (25, 3)

def smpl_to_torch(smpl_param):
    for key in smpl_param.keys():
        smpl_param[key] = torch.from_numpy(smpl_param[key]).to(torch.float32).reshape(1, -1)
    return smpl_param


def read_zju_smpl(root, pid):
    """
    Read SMPL parameters from ZJU dataset
    input: json root, frame, pid
    output: {'betas':[1, 3], 
             'body_pose':[1, 69],
             'global_orient':[1, 3],
             'transl':[1, 3]}
    """
    data = read_json(root)[pid]
    output = {}
    output['body_pose'] = data['poses']
    output['betas'] = data['shapes']
    output['global_orient'] = data['Rh']
    output['transl'] = data['Th']
    for key in output.keys():
        output[key] = np.array(output[key], dtype=np.float32)
    output['body_pose'] = output['body_pose'][:, 3:]
    return output

def read_smpl(root, pid):
    """
    Read SMPL parameters from Hi4D smpl dataset
    input: npz root, frame, pid
    output: {'betas':[1, 3], 
             'body_pose':[1, 69],
             'global_orient':[1, 3],
             'transl':[1, 3]}
    """
    output = {}
    name = f'human_{pid}'
    smpl_params = dict(np.load(root, allow_pickle=True))
    for k, v in smpl_params[name].item().items():
        output[k] = np.array(v, dtype=np.float32)
    return output

def read_smpl_frame(root, pid, frame_idx):
    smpl_params = read_smpl(root, pid)
    smpl_params = get_smpl_frame(smpl_params, frame_idx)
    return smpl_params

def get_smpl_frame(smpl_params, frame_idx):
    smpl_param = {}
    for key in ['betas', 'body_pose', 'global_orient', 'transl']:
        smpl_param[key] = smpl_params[key][frame_idx]
    return smpl_param

def smpl_np_to_torch(smpl_params):
    for key in ['betas', 'body_pose', 'global_orient', 'transl']:
        smpl_params[key] = torch.from_numpy(smpl_params[key]).to(torch.float32).reshape(1, -1)
    return smpl_params

def smpl_torch_to_np(smpl_params):
    for key in ['betas', 'body_pose', 'global_orient', 'transl']:
        smpl_params[key] = smpl_params[key].detach().cpu().numpy().reshape(1, -1)
    return smpl_params


def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.05, MIN_PIXEL=5):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh
    if valid.sum() < 3:
        return [0, 0, 100, 100, 0]
    valid_keypoints = keypoints[valid][:,:-1]
    center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))/2
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    if bbox_size[0] < MIN_PIXEL or bbox_size[1] < MIN_PIXEL:
        return [0, 0, 100, 100, 0]
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0]/2, 
        center[1] - bbox_size[1]/2,
        center[0] + bbox_size[0]/2, 
        center[1] + bbox_size[1]/2,
        keypoints[valid, 2].mean()
    ]
    bbox = np.array(bbox).tolist()
    return bbox

def vertices_projection(verts, R, T, K):
    x_c = verts @ R.T + T.T
    x_p_homo = x_c @K.T
    x_p = x_p_homo[:, :2]/x_p_homo[:, 2:]
    pixel = x_p.round().astype(int)
    bbox = [np.min(pixel[:, 0]), np.min(pixel[:, 1]), np.max(pixel[:, 0]), np.max(pixel[:, 1])]
    return pixel, bbox

def keypoint_projection(P, keypoints3d):
    if keypoints3d.shape[1] == 3:
        keypoints3d = np.hstack((keypoints3d, np.ones((keypoints3d.shape[0], 1))))
    kcam = np.hstack([keypoints3d[:, :3], np.ones((keypoints3d.shape[0], 1))]) @ P.T
    kcam = kcam[:, :]/kcam[:, 2:]
    k2d = np.hstack((kcam, (keypoints3d[:, 3:]>0.)&(kcam[:, 2:] >0.1)))
    bbox = bbox_from_keypoints(k2d)
    return k2d, bbox

def get_vertices_from_json(root, pid):
    verts = read_json(root)
    verts = np.array(verts[pid]['vertices'], dtype=np.float32)
    return verts

class SMPLKeypoints:    
    def __init__(self,
                 root = './Hi4D_AvatarPose/pair14_hug14',
                 smpl_root = 'smpl_gt/smpl.npz',
                 pids = [0, 1],
                 frames = [1, 101, 1],
                 out_root = './4d_association/data/hi4d_14_hug14',
                 out_name = 'gt.txt'):
        self.root = root
        self.smpl_root = smpl_root
        smpl_name = os.path.join(self.root, self.smpl_root)
        self.smpl_params = np.load(smpl_name, allow_pickle=True)
        self.model_path = 'data/SMPLX/smpl'
        self.gender = 'neutral'
        self.body_model = SMPL(self.model_path, gender=self.gender)
        self.pids = pids
        self.frames = np.arange(frames[0], frames[1], frames[2])
        self.keys = ['betas', 'global_orient', 'body_pose', 'transl']
        self.out_root = out_root
        self.out_name = out_name
        
    def get_smpl_params(self, pid, frame_idx):
        name = f'human_{pid}'
        all_params = self.smpl_params[name].item()
        smpl_param = {}
        for key in self.keys:
            smpl_param[key] = all_params[key][frame_idx] # all_params[key]: (n_frames, 1, n_dim)
        return smpl_param
    
    def get_model_keypoints(self, pid, frame_idx, model_type='body25_vertices'):
        smpl_param = self.get_smpl_params(pid, frame_idx) # smpl_param[key]: (1, n_dim)
        smpl_param_torch = smpl_to_torch(smpl_param) # smpl_param_torch[key]: (1, n_dim)
        smpl_output = self.body_model(
            betas=smpl_param_torch['betas'],
            body_pose=smpl_param_torch['body_pose'],
            global_orient=smpl_param_torch['global_orient'],
            transl=smpl_param_torch['transl'],
        )
        if model_type == 'body25_vertices':
            keypoints = smplvertices_to_body25(smpl_output.vertices.detach().cpu().numpy())
        elif model_type == 'body25_joints':
            keypoints = smpljoints_to_modeljoints(smpl_output.joints.detach().cpu().numpy(), model_type='body25_joints')
        elif model_type == 'shelf15':
            keypoints = smpljoints_to_modeljoints(smpl_output.joints.detach().cpu().numpy(), model_type='shelf15')
        else:
            raise ValueError(f'from_type {model_type} not supported')
        return keypoints #(n_joints, 3)
    
    def get_keypoints_all(self, func):
        keypoints_list_all = []
        for frame_idx in tqdm(range(len(self.frames))):
            keypoints_list = []
            for pid in self.pids:
                keypoints = func(pid, frame_idx)
                keypoints_dict = {'id': pid, 'keypoints3d': keypoints}
                keypoints_list.append(keypoints_dict)
            keypoints_list_all.append(keypoints_list)
        return keypoints_list_all
    
    
    def save_keypoints(self, func,  out_type='npz'):
        os.makedirs(self.out_root, exist_ok=True)
        keypoints_list_all = self.get_keypoints_all(func)
        if out_type == 'npz':
            np.savez(os.path.join(self.out_root, self.out_name), keypoints_list_all)
        if out_type == 'txt':
            write_keypoints3d_txt(os.path.join(self.out_root, self.out_name), keypoints_list_all)
        if out_type == 'json':
            for frame_idx in range(len(self.frames)):
                frame = self.frames[frame_idx]
                keypoints_list = keypoints_list_all[frame_idx]
                name = os.path.join(self.out_root, f'{frame:06d}.json')
                write_keypoints3d(name, keypoints_list)
            
    def read_keypoints_npz(self, pid, frame_idx):
        datas = np.load(os.path.join(self.out_root, self.out_name), allow_pickle=True)
        datas = datas['arr_0']
        keypoints = datas[frame_idx][pid]['keypoints3d']
        return keypoints #(n_joints, 3)