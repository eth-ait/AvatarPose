from AvatarPose.vis.vis_pose import vis_kpts
from AvatarPose.utils.cam_utils import read_cameras
import os
import cv2
from os.path import join
from glob import glob
import numpy as np
import torch
from AvatarPose.vis.vis_video import combine_video
from AvatarPose.utils.file_utils import read_json

def get_smpl(smpl_name):
    smpl_params = dict(np.load(smpl_name, allow_pickle=True))
    smpl_params_new = {}
    for name in smpl_params.keys():
        assert('frames' in smpl_params_new[name]), 'There is no frames in these smpl parameters'
        smpl_params_new[name]['frames'] = list(smpl_params_new[name]['frames'])
    return smpl_params_new

def retrieve_smpl_init(smpl_params, frame):
    keys = ['betas', 'body_pose', 'global_orient', 'transl']
    smpl_init = {}
    for name in smpl_params.keys():
        if frame in smpl_params[name]['frames']:
            idx = smpl_params[name]['frames'].index(frame)
            smpl_param = {k: torch.from_numpy(smpl_params[name][k][idx].copy()) for k in keys}
            smpl_init[name] = smpl_param
    return smpl_init

def get_masks_gt(mask_root, frame):
    assert(mask_root), "mask for all objs does not exist"
    maskname = join(mask_root, f"{frame:06d}.png")
    mask = cv2.imread(maskname, 0)
    pids_mask = mask != 0
    return pids_mask

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


def draw_kpts(root, dataname, cam, kpts_1_root, kpts_2_root, vis_root, ext):
    root_data = join(root, dataname)
    image_root = join(root_data, 'images', cam)
    cams_dict = read_cameras(root_data)
    cam_info = cams_dict[cam]
    imgnames = sorted(glob(join(image_root, '*.jpg')))

    for i, imgname in enumerate(imgnames):
        basename = os.path.basename(imgname)
        frame = int(basename.split('.')[0])
        print(frame)
        img = cv2.imread(imgname)
        img = img.astype(np.uint8)
        img_1 = img.copy()
        img_2 = img.copy()
        
        kpts_1_dict = get_keypoints3d(join(kpts_1_root, f'{frame:06d}.json'))
        kpts_2_dict = get_keypoints3d(join(kpts_2_root, f'{frame:06d}.json'))       
        
        img_1 = vis_kpts(kpts_1_dict, cam_info['P'], img_1)
        img_2 = vis_kpts(kpts_2_dict, cam_info['P'], img_2)
        
        os.makedirs(f"{vis_root}/{dataname}/{ext}/{cam}", exist_ok=True)
        kpts_img = np.concatenate([img_1, img_2], axis=1)
        cv2.imwrite(f'{vis_root}/{dataname}/{ext}/{cam}/{dataname}_{cam}_{frame:06d}.png', kpts_img)
    combine_video(f'{vis_root}/{dataname}/{ext}/{cam}', f'{vis_root}/{dataname}', f'{ext}_{dataname}_{cam}')
    
    
        
        

    
if __name__ == '__main__':
    root = '/home/username/Hi4D_AvatarPose' # image root
    dataname = 'pair14_talk14' 
    kpts_1_root = f'/home/username/Hi4D_AvatarPose/{dataname}/skel19_gt'
    kpts_2_root = '/home/username/outputs/hi4d/exp/14_talk14/kpts_est_kpts_opt_1'
    cam = '52'
    ext = 'gt_opt'
    vis_root = '/home/username/outputs/vis'
    draw_kpts(root, dataname, cam, kpts_1_root, kpts_2_root, vis_root, ext)

