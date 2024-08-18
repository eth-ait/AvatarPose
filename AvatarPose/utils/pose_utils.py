import numpy as np
import torch
import hydra
from os.path import join
from AvatarPose.utils.file_utils import write_keypoints3d
from AvatarPose.utils.file_utils import read_json
from instant_avatar.deformers.smplx import SMPL
np.random.seed(2023)


model_path=hydra.utils.to_absolute_path('data/SMPLX/smpl')
gender='neutral'
body_model = SMPL(model_path, gender=gender)
regressor = np.load(hydra.utils.to_absolute_path('data/SMPLX/J_regressor_body25.npy'))

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

def smpl_to_keypoints(smpl_params):
    output = body_model(
            beta=smpl_params['betas'],
            body_pose=smpl_params['body_pose'],
            global_orient=smpl_params['global_orient'],
            transl=smpl_params['transl']
        )
    joints = np.matmul(regressor, output.vertices.detach().cpu().numpy().reshape(-1, 3))
    return joints


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

def get_keypoints3d(root, pid):
    keypoints3d = read_json(root)[pid]['keypoints3d']
    keypoints3d = np.array(keypoints3d, dtype=np.float32)
    return keypoints3d