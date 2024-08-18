import os
import shutil
import numpy as np
import cv2
import open3d as o3d
from os.path import join
from glob import glob
from tqdm import tqdm
from AvatarPose.utils.file_utils import write_keypoints3d, write_smpl_all
from AvatarPose.utils.cam_utils import write_camera
from AvatarPose.vis.vis_video import combine_videos
from AvatarPose.utils.utils import transform_smpl
from AvatarPose.utils.keypoints_utils import body25_to_skel19
from instant_avatar.deformers.smplx import SMPL



def get_translation_rotation(root):
    # read mesh
    meshanme = sorted(glob(join(root, 'frames', '*.obj')))[0]
    mesh = o3d.io.read_triangle_mesh(meshanme)
    vertices = np.asarray(mesh.vertices)

    # calculate global rotation and translation
    R_global = cv2.Rodrigues(np.array([np.pi/2, 0, 0]))[0]
    vertices_R = vertices @ R_global.T
    z_min = np.min(vertices_R[:, 2])
    T_global = np.array([0, 0, -z_min]).reshape(3, 1)
    
    return R_global, T_global

def save_camera(root, outroot, R_global, T_global):
    # read camera
    cameraname = join(root, 'cameras', 'rgb_cameras.npz')
    cameras = dict(np.load(cameraname))
    cameras_out = {}
    for i, cam in enumerate(cameras['ids']):
        K = cameras['intrinsics'][i]
        dist = cameras['dist_coeffs'][i:i+1]
        RT = cameras['extrinsics'][i]
        R = RT[:3, :3]
        T = RT[:3, 3:]
        cameras_out[str(cam)] = {
            'K': K,
            'dist': dist,
            'R': R,
            'T': T
        }
        center = - R.T @ T
        print(cam, center.T[0])
    cameras = cameras_out


    # rotate and translate cameras
    for key, cam in cameras.items():
        cam['R'] = cam['R'] @ R_global.T
        cam.pop('Rvec', '')
        center = - cam['R'].T @ cam['T']
        newcenter = center + T_global
        newT = -cam['R'] @ newcenter
        cam['T'] = newT
        center = - cam['R'].T @ cam['T']
        print(center.T)

    # write cameras and mesh
    write_camera(cameras, outroot)


def save_keypoints_gt(root, outroot, regressor, R_global, T_global):
    filenames = sorted(glob(join(root, 'smpl', '*.npz')))

    smpl_all = {}
    for pid in range(2):
        name = f'human_{pid}'
        smpl_all[name] = {}
        for key in ['betas', 'body_pose', 'global_orient', 'transl', 'frames']:
            smpl_all[name][key] = []
        
    for filename in tqdm(filenames):
        data = dict(np.load(filename))
        # rotate and translate vertices
        vertices = data['verts']
        vertices = vertices @ R_global.T + T_global.T
        # calculate body25 joints
        joints = np.matmul(regressor[None], vertices)
        
        joints_list = []
        for pid in range(len(joints)):
            # calculate skel19 joints
            skel19 = body25_to_skel19(joints[pid])
            joints_dict = {
                'id': pid,
                'keypoints3d': skel19
            }
            joints_list.append(joints_dict)
        
        outname_joint = join(outroot, 'skel19_gt', os.path.basename(filename).replace('.npz', '.json'))
        write_keypoints3d(outname_joint, joints_list)
                    
        # calculate smpl
        joints_list = []
        for pid in range(len(joints)):
            name = f'human_{pid}'
            joints_3d = data['joints_3d'][pid]
            joint_0 = joints_3d[:1].reshape(1, -1)
            smpl_params = {
                'betas': data['betas'][pid].reshape(1, -1),
                'global_orient': data['global_orient'][pid].reshape(1, -1),
                'body_pose': data['body_pose'][pid].reshape(1, -1),
                'transl': data['transl'][pid].reshape(1, -1)
            }
            smpl_new = transform_smpl(smpl_params, joint_0, R_global, T_global)
            

            
            for key in smpl_new.keys():
                smpl_all[name][key].append(smpl_new[key])
            smpl_all[name]['frames'].append(int(os.path.basename(filename).split('.')[0]))


def save_smpl_gt(root, outroot, R_global, T_global):
    filenames = sorted(glob(join(root, 'smpl', '*.npz')))

    smpl_all = {}
    for pid in range(2):
        name = f'human_{pid}'
        smpl_all[name] = {}
        for key in ['betas', 'body_pose', 'global_orient', 'transl', 'frames']:
            smpl_all[name][key] = []
        
    for filename in tqdm(filenames):
        data = dict(np.load(filename))
                    
        # calculate smpl
        for pid in range(len(data['body_pose'])):
            name = f'human_{pid}'
            joints_3d = data['joints_3d'][pid]
            joint_0 = joints_3d[:1].reshape(1, -1)
            smpl_params = {
                'betas': data['betas'][pid].reshape(1, -1),
                'global_orient': data['global_orient'][pid].reshape(1, -1),
                'body_pose': data['body_pose'][pid].reshape(1, -1),
                'transl': data['transl'][pid].reshape(1, -1)
            }
            smpl_new = transform_smpl(smpl_params, joint_0, R_global, T_global)
            
            for key in smpl_new.keys():
                smpl_all[name][key].append(smpl_new[key])
            smpl_all[name]['frames'].append(int(os.path.basename(filename).split('.')[0]))
    
    for pid in range(2): 
        name = f'human_{pid}'       
        for key in ['betas', 'global_orient', 'body_pose', 'transl']:
            smpl_all[name][key] = np.array(smpl_all[name][key], dtype=np.float32)
    
    output_path = join(outroot, 'smpl_gt')
    write_smpl_all(output_path, smpl_all) 
        

def save_images(root, outroot):
    if not os.path.exists(join(outroot, 'images')):
        shutil.copytree(join(root, 'images'), join(outroot, 'images'))
        
def save_videos(outroot):
    # extract videos
    image_root = join(outroot, 'images')
    video_root = join(outroot, 'videos')
    combine_videos(image_root, video_root)

def save_masks_gt(root, outroot):
    # copy gt mask
    if not os.path.exists(f'{outroot}/mask_gt'):
        cmd = f'cp -r {root}/seg/img_seg_mask/ {outroot}/'
        os.system(cmd)
        cmd = f'mv {outroot}/img_seg_mask {outroot}/mask_gt'
        os.system(cmd)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='./Hi4D')
    parser.add_argument('--outdir', type=str, default='./Hi4D_AvatarPose')
    parser.add_argument('--seq', type=str, default='pair14/talk14')
    args = parser.parse_args()

    # Input and output path
    database = args.indir
    outdatabase = args.outdir
    seq = args.seq
    regressor = np.load('data/SMPLX/J_regressor_body25.npy')
    root = join(database, seq)
    outroot = join(outdatabase, seq.replace('/', '_'))
    
    R_global, T_global = get_translation_rotation(root)
    save_camera(root, outroot, R_global, T_global)
    save_keypoints_gt(root, outroot, regressor, R_global, T_global)
    save_smpl_gt(root, outroot, R_global, T_global)
    save_images(root, outroot)
    save_videos(outroot)
    save_masks_gt(root, outroot)
