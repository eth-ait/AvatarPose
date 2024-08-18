
import argparse
from concurrent.futures import process
from glob import glob
import json
import numpy as np
import os
import trimesh
from instant_avatar.deformers.smplx import SMPL
from AvatarPose.utils.cam_utils import read_cameras
from AvatarPose.utils.file_utils import read_json
import hydra
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.billboard import Billboard
from aitviewer.viewer import Viewer
from pathlib import Path
import cv2
import torch

def get_keypoints3d(root):
    #{pid: keypoints3d, pid: keypoints3d}
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
# smpl model
model_path = 'data/SMPLX/smpl'
gender = 'neutral'
body_model = SMPL(model_path, gender=gender)
faces = body_model.faces
limbs = np.array([[-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [1, 6], [2, 7], [3, 8], [4, 9], [4, 10],
         [5, 11], [6, 12], [7, 13], [8, 14], [11, 15], [12, 16], [13, 18], [14, 17]])
# Display in viewer
v = Viewer()
v.scene.camera.position = [0, 1.2, 4]
v.scene.camera.target = [0, 1, 0]
v.shadows_enabled = True   
v.auto_set_camera_target = False 
v.auto_set_floor = False
rotation = cv2.Rodrigues(np.array([-np.pi/2, 0, 0], dtype=np.float32))[0]
v.scene.lights[1].elevation = -50
colors = [(0.11,0.639,1.0,1), (1.0,0.471,0.11,1),  (0.6, 0.61, 0.6, 1.0), ]
colors_skel = [(0.11,0.639,1.0,1.0), (1.0,0.471,0.11,1.0), (1.0, 0.0, 0.0,1.0)]


@hydra.main(config_path="./confs", config_name="opt_avatar_smpl")
def main(opt):
    # Load predicted SMPLs
    root = opt.dataset.opt.root
    data = f'pair{opt.dataset.subject}'
    frames = opt.dataset.split_opt.val.ranges
    start_frame = opt.dataset.opt.start_frame
    smpl_path = f'smpl_{opt.model.opt.vis_log_name}/smpl_all.npz'
    joint_path = f'kpts_{opt.model.opt.vis_log_name}'

    data_root = os.path.join(root, data)
    cams_info = read_cameras(data_root)
    H = 1280
    W = 940
    for key in cams_info.keys():
        K = np.array(cams_info[key]['K']).reshape(-1, 3)
        K_new = K.copy()
        RT = np.array(cams_info[key]['RT']).reshape(3, 4)
        RT[:3, :3] = RT[:3, :3]@rotation.T
        cam = OpenCVCamera(K_new, RT[:3], W, H, viewer=v)
        v.scene.add(cam)
        v.scene.camera = cam


    smpl_params = dict(np.load(smpl_path, allow_pickle=True))


    for name in smpl_params.keys():
        pid = int(name.split('_')[-1])
        smpl_params[name] = smpl_params[name].item()
        smpl = {}
        frames_smpl = smpl_params[name]['frames']
        frames_smpl = list(frames_smpl)
        verts_list = []
        faces_list = []
        joints_list = []
        for frame in range(frames[0], frames[1], frames[2]):
            assert(frame in frames_smpl)
            
            idx = frames_smpl.index(frame)
            time = frame - start_frame
            for key in ['betas', 'body_pose', 'global_orient', 'transl']:
                smpl[key] = smpl_params[name][key][idx]
            
            
            smpl_torch = {key: torch.from_numpy(smpl[key]).to(torch.float32) for key in smpl.keys()}
            output = body_model(**smpl_torch)
            verts = output.vertices.detach().cpu().numpy()[0]
            verts = verts@rotation.T
            verts_list.append(verts)
            faces_list.append(faces)
            joints_dict = get_keypoints3d(os.path.join(joint_path, f'{frame:06d}.json'))
            joints = joints_dict[name]['kpts'][0][:, :3]
            joints = (joints @ rotation.T)
            joints_list.append(joints)
        
        skeleton = Skeletons(np.array(joints_list), limbs, radius= 0.02, gui_affine=False, color = colors_skel[pid]) 
        v.scene.add(skeleton) 
        meshes = VariableTopologyMeshes(verts_list, faces_list, color=colors[pid], preload=False)       
        v.scene.add(meshes)
    v.run()
    
    
if __name__ == "__main__":
    main()

