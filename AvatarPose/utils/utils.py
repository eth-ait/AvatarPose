import torch
import numpy as np
import nvidia_smi
from omegaconf.dictconfig import DictConfig
import cv2

def check_max_epoch(opt):
    max_epoch = 0
    for key in opt.model.opt.stages.keys():
        vals = opt.model.opt.stages[key]
        for val in vals:
            if val[1] > max_epoch:
                max_epoch = val[1]
    assert (max_epoch == opt.train.max_epochs), "opt.train.max_epochs not equal to the max epochs required"

def find_dict_diff(d1, d0, path=""):
    for k in d1:
        if k in d0:
            if type(d1[k]) is DictConfig:
                find_dict_diff(d1[k], d0[k], "%s.%s" % (path, k) if path else k)
            elif d1[k] != d0[k]:
                result = f"{path}.{k}: {d1[k]} or {d0[k]} ?"
                print(result)
        else:
            print (f"{path}.{k} not exist in the existing config")
            
def to_np(val):
    if isinstance(val, dict):
        for key in val.keys():
            val[key] = to_np(val[key])
        return val
    elif isinstance(val, torch.Tensor):
        if val.is_cpu:
            return val.numpy()
        elif val.is_cuda:
            return val.detach().cpu().numpy()
    elif isinstance(val, list):
        return [to_np(v) for v in val] 
    else:
        return val 
    
def to_tensor(val):
    if isinstance(val, dict):
        for key in val.keys():
            val[key] = to_tensor(val[key])
        return val
    elif isinstance(val, np.ndarray):
        return torch.from_numpy(val)
    elif isinstance(val, torch.Tensor) and val.is_cuda:
        return val.detach().cpu()
    elif isinstance(val, list):
        return [to_tensor(v) for v in val] 
    else:
        return val

def to_device(val, device):
    if isinstance(val, dict):
        for key in val.keys():
            val[key] = to_device(val[key], device)
        return val
    elif isinstance(val, torch.Tensor):
        return val.to(device)
    elif isinstance(val, np.ndarray):
        return torch.from_numpy(val).to(device)
    elif isinstance(val, list):
        return [to_device(v, device) for v in val] 
    else:
        return val 
    

def get_gpu_memory():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # in GB
    total = info.total / 1024**3
    used = info.used / 1024**3
    nvidia_smi.nvmlShutdown()
    return total, used

def transform_smpl(smpl_params, joints_0, R_global, T_global):
    """_summary_

    Args:
        smpl_params (dict): smpl_params: body_pose, betas, global_orient, transl
        joint_0 (np.array, [1, 3]): joint 0 pelvis location with smpl_params['Th'] translation
        R_global (np.array, [3, 3]): global rotation
        T_global (np.array, [3, 1]): global translation

    Returns:
        dict: new smpl_params
    """
    R_0 = cv2.Rodrigues(smpl_params['global_orient'])[0]
    T_0 = smpl_params['transl'].reshape(-1, 1)
    Rnew = R_global @ R_0
    joint_0 = joints_0.reshape(-1, 1)-T_0 # joint 0 with not translation
    T_z = T_0 + joint_0 - R_0 @ joint_0
    T_z_new = R_global@T_z + T_global
    Tnew = T_z_new -joint_0 + Rnew @ joint_0
    smpl_new = smpl_params.copy()
    smpl_new['global_orient'] = cv2.Rodrigues(Rnew)[0].reshape(1, 3)
    smpl_new['transl'] = Tnew.reshape(1, 3)
    return smpl_new