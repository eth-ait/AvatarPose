import os
from os.path import join
import cv2
import copy
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import hydra
from glob import glob
import torch
from torch.utils.data import DataLoader
from AvatarPose.utils.cam_utils import read_cameras
from AvatarPose.utils.utils_readdata import scale_and_undistort
from AvatarPose.utils.utils_sampling import get_rays
from AvatarPose.vis.vis_pose import get_keypoints3d


class BaseDataset:
    def __init__(self, root, split, opt, split_opt) -> None:
        self.root = root
        self.split = split
        self.split_opt = split_opt
        self.start_frame = opt.start_frame
        self.ranges = split_opt.ranges
        self.vis_dataset = opt.vis_dataset
        self.camnames = self.get_camnames(split_opt.camnames)
        self.cams_info = read_cameras(root)
        # get meta information
        if self.split != 'test':
            self.infos = self.get_infos(root, self.cams_info, self.ranges, opt.image_args)
        if self.split == 'test':
            self.camargs = opt.camera_args
            self.demoargs = opt.demo_args
            self.infos = self.create_demo_cameras(opt.image_args.nv_scale, opt.camera_args, opt.demo_args)
        self.image_args = opt.image_args
        self.mask_args = opt.mask_args
        self.smpl_init = self.get_smpl(join(self.root, opt.smpl_args.init_root, 'smpl.npz'))
        self.kpts_init_list = self.get_kpts(join(self.root, opt.keypoints3d_args.init_root))
        if self.split == 'val':
            self.smpl_gt = self.get_smpl(join(self.root, opt.smpl_args.gt_root, 'smpl.npz'))
            self.kpts_gt_list = self.get_kpts(join(self.root, opt.keypoints3d_args.gt_root))
        self.ray_o_d_cache = {}
        self.img_size = (opt.img_size[0], opt.img_size[1])
        if 'train' in self.split:
            self.sampler = hydra.utils.instantiate(split_opt.sampler)
        self.coords = self.get_coords() 
        self.check_data()
        
    def check_data(self):
        
        if self.split=='train_smpl':
            visited = set()
            from tqdm import tqdm
            for i in tqdm(range(len(self)), desc='check data'):
                info = self.infos[i]
                cam = info['cam']
                frame = info['frame']
                if cam in visited: continue
                if frame != 50: continue
                visited.add(cam)
                data=self[i]
    
    def get_smpl(self, smpl_name):
        smpl_params = dict(np.load(smpl_name, allow_pickle=True))
        for name in smpl_params.keys():
            smpl_params[name] = smpl_params[name].item()
            assert(smpl_params[name].get('frames')), 'There is no frames in these smpl parameters'
            smpl_params[name]['times'] = [(val - self.start_frame) for val in smpl_params[name]['frames']]
        return smpl_params
    
    def get_kpts(self, kpts_root):
        kpts_names = sorted(glob(join(kpts_root, '*.json')))
        kpts_list = []
        for kpts_name in kpts_names:
            kpts_list.append(get_keypoints3d(kpts_name))
        return kpts_list # [{pid:kpts}]
                
        
    def retrieve_smpl(self, name):
        keys = ['betas', 'body_pose', 'global_orient', 'transl']
        smpl_param = {k: torch.from_numpy(self.smpl_init[name][k].copy()) for k in keys}
        smpl_param['times'] = self.smpl_init[name]['times']
        return smpl_param
    
    def retrieve_smpl_init(self, time_idx):
        keys = ['betas', 'body_pose', 'global_orient', 'transl']
        smpl_init = {}
        for name in self.smpl_init.keys():
            if time_idx in self.smpl_init[name]['times']:
                idx = self.smpl_init[name]['times'].index(time_idx)
                smpl_param = {k: torch.from_numpy(self.smpl_init[name][k][idx].copy()) for k in keys}
                smpl_init[name] = smpl_param
        return smpl_init
    
    def retrieve_smpl_gt(self, time_idx):
        keys = ['betas', 'body_pose', 'global_orient', 'transl']
        smpl_gt = {}
        for name in self.smpl_gt.keys():
            if time_idx in self.smpl_gt[name]['times']:
                idx = self.smpl_init[name]['times'].index(time_idx)
                smpl_param = {k: torch.from_numpy(self.smpl_gt[name][k][idx].copy()) for k in keys}
                smpl_gt[name] = smpl_param
        return smpl_gt
    
    def retrieve_kpts_init(self, time_idx):
        kpts_init = self.kpts_init_list[time_idx]
        kpts_init_new = {}
        for name in kpts_init.keys():
            kpts_init_new[name] = {}
            kpts_init_new[name]['kpts'] = torch.from_numpy(kpts_init[name])
        return kpts_init_new
    
    def retrieve_kpts_gt(self, time_idx):
        kpts_gt = self.kpts_gt_list[time_idx]
        kpts_gt_new = {}
        for name in kpts_gt.keys():
            kpts_gt_new[name] = {}
            kpts_gt_new[name]['kpts'] = torch.from_numpy(kpts_gt[name])
        return kpts_gt_new
        
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, idx):
        assert(self.split in ['train_avatar', 'train_avatar_refine', 'train_smpl', 'train_arm', 'val', 'test'])
        info = copy.deepcopy(self.infos[idx])
        img_orig, scale = self.get_img(info)
        objs, names_all = self.get_objects(info)
        objs, pids_mask, sample_mask = self.get_masks(objs, info)
        bg_color, img = self.set_bg_color(pids_mask, img_orig)
        info = self.update_info(info, scale)
        rays_o, rays_d = self.get_rays(info)
        ret = self.sample_objs(objs, img, rays_o, rays_d, bg_color, pids_mask, sample_mask)
        meta = {
            'split': self.split,
            'H': self.img_size[0],
            'W': self.img_size[1],
            'index': idx,
            'cam': info['cam'],
            'cam_idx': info['cam_idx'],
            'frame_idx': info['frame_idx'],
            'frame': info['frame'],
            'time_idx': info['time_idx'],
            'names': list(names_all),
        }
        ret['cam_info'] = info['cam_info']
        ret['meta'] = meta
        if self.vis_dataset:
            self.vis_everything(ret, img_orig, objs, img, pids_mask)
        
        return ret
    
    def set_bg_color(self, pids_mask, img):
        mask_all = pids_mask > 0
        if 'train' in self.split:
            # 不在人体的部分是random color
            if self.image_args.train_white_bkgd:
                bg_color = np.ones_like(img).astype(np.float32)
            else:
                bg_color = np.random.rand(*img.shape).astype(np.float32)
            img = img * mask_all[..., None] + (1 - mask_all[..., None]) * bg_color
            img = img.astype(np.float32)
        elif self.split == "val":
            # test的时候不在人体的部分是白色
            bg_color = np.ones_like(img).astype(np.float32)
            img = img * mask_all[..., None] + (1 - mask_all[..., None]) * bg_color
            img = img.astype(np.float32)
        else:
            bg_color = np.ones_like(img).astype(np.float32)
        return bg_color, img
    
    def get_rays(self, info):
        H, W = self.img_size
        cam_info = info['cam_info']
        cam = info['cam']
        if cam not in self.ray_o_d_cache.keys():
            rays_o, rays_d = get_rays(H, W, cam_info)
            self.ray_o_d_cache[cam] = (rays_o, rays_d)
        rays_o, rays_d = self.ray_o_d_cache[cam]
        return rays_o, rays_d   
        
    
    def get_objects(self, info):
        time_idx = info['time_idx']
        # determine all the objects in this frame
        names_all = [name for name, smpl in self.smpl_init.items() if time_idx in smpl['times']]
        # if self.split == 'val':
            # for obj_key in self.split_opt.object_keys:
            #     assert(obj_key in names_all), 'One validation object is not in the frame'
            # names_all = self.split_opt.object_keys
        # elif self.split == 'test':
        #     for obj_key in info['object_keys']:
        #         assert(obj_key in names_all), 'One test object is not in the frame'
        #     names_all = info['object_keys']
        
        assert time_idx < len(self.kpts_init_list), "time_idx is out of range (check dataset.opt.start_frame?)"
        
        # get all the objects in the frame
        objs = {} # {name: obj} all information about all humans
        for name in names_all:
            pid = int(name.split('_')[-1])
            smpl = self.smpl_init[name]
            smpl_idx = smpl['times'].index(time_idx)
            obj = {
                'pid': pid,
                'betas': np.array(smpl['betas'][smpl_idx], dtype=np.float32).reshape(-1),
                'body_pose': np.array(smpl['body_pose'][smpl_idx], dtype=np.float32).reshape(-1),
                'global_orient': np.array(smpl['global_orient'][smpl_idx], dtype=np.float32).reshape(-1),
                'transl': np.array(smpl['transl'][smpl_idx], dtype=np.float32).reshape(-1),
                'kpts_init': self.kpts_init_list[time_idx][name],
            }
            objs[name] = obj
        return objs, names_all
    
    def get_masks(self, objs, info):
        if self.split == 'train_arm':
            return self.get_arm_masks(objs, info)
        else:
            return self.get_masks_human(objs, info) 
        
    def get_masks_human(self, objs, info):
        cam = info['cam']
        cam_info = info['cam_info']
        frame = info['frame']
        time_idx = info['time_idx']
        pids_mask = np.zeros(self.img_size) # mask for all humans with pid+1 stored
        if self.split != 'test':
            masks_root = join(self.root, self.mask_args.root, cam)
            if self.mask_args.per_obj:
                Warning("note if it is gt mask, the pid might not be the same with the smpl_init pid")
                for name, obj in objs.items():
                    pid = obj['pid']
                    maskname = join(masks_root, str(pid), f"{frame:06d}.png")
                    assert(os.path.exists(maskname)), "mask for the obj does not exist"
                    mask = cv2.imread(maskname, 0)
                    mask, scale = scale_and_undistort(mask, cam_info, cam, undistort=self.mask_args.undis, res=self.img_size)
                    obj['mask'] = mask != 0
                    if obj['mask'].sum() > 0:
                        pids_mask[obj['mask']] = pid + 1
            else:
                all_mask_path = join(masks_root, 'all')
                assert(os.path.exists(all_mask_path)), "mask path for all objs does not exist"
                maskname = join(all_mask_path, f"{time_idx}.png")
                if not os.path.exists(maskname):
                    maskname = join(all_mask_path, f"{frame:06d}.png")
                assert(os.path.exists(maskname)), f"{maskname} mask file for all objs does not exist"
                mask = cv2.imread(maskname, 0)
                mask, scale = scale_and_undistort(mask, cam_info, cam, undistort=self.mask_args.undis, res=self.img_size)
                pids_mask = mask != 0
        return objs, pids_mask, pids_mask
    
    def get_arm_masks(self, objs, info):
        cam = info['cam']
        frame = info['frame']
        time_idx = info['time_idx']
        pids_mask = np.zeros(self.img_size) # mask for all humans with pid+1 stored
        mask_arm_path = join(self.root, self.mask_args.mask_arm_root, cam, 'all')
        mask_path = join(self.root, self.mask_args.root, cam, 'all')
        if self.split != 'test':
            # print("use all arm masks")
            assert(os.path.exists(mask_arm_path)), "arm mask path for all objs does not exist"
            maskname = join(mask_arm_path, f"{frame:06d}.png")
            mask = cv2.imread(maskname, 0)
            sample_mask = mask != 0
            
            assert(os.path.exists(mask_path)), "mask for all objs does not exist"
            maskname = join(mask_path, f"{time_idx}.png")
            if not os.path.exists(maskname):
                maskname = join(mask_path, f"{frame:06d}.png")
            mask = cv2.imread(maskname, 0)
            pids_mask = mask != 0
        return objs, pids_mask, sample_mask
    
    def sample_objs(self, objs, img, rays_o, rays_d, bg_color, pids_mask, sample_mask):
        # (..., 3), (...), (...), (..., 3), (..., 3), (..., 3), (..., 2)
        img_keys = ['rgb', 'pids_mask', 'alpha', 'bg_color', 'rays_o', 'rays_d', 'coords']
        # (10, ), (69, ), (3, ), (3, ), (19, 3)
        obj_keys = ['betas', 'body_pose', 'global_orient', 'transl', 'kpts_init']
        datums = {}
        if 'train' in self.split:
            for key in img_keys:
                datums[key] = []
            for name, obj in objs.items():
                mask_temp = None
                # extract img samples using masks
                if 'mask' not in obj and sample_mask.sum() > 20:
                    mask_temp = sample_mask.astype(np.float32)
                elif 'mask' in obj and obj['mask'].sum() > 20:
                    mask_temp = obj['mask'].astype(np.float32)
                
                if mask_temp is not None and mask_temp.sum() > 20:
                    # mask_obj: (sample_shapes, ), pids_mask_obj: (sample_shapes, ), coords_obj: (sample_shapes, 2), rgb_obj: (sample_shapes, 3), ...
                    (mask_obj, rgb_obj, rays_o_obj, rays_d_obj, bg_color_obj, coords_obj, pids_mask_obj) = \
                        self.sampler.sample(mask_temp, img, rays_o, rays_d, bg_color, self.coords, pids_mask) 
                    if pids_mask_obj.ndim == coords_obj.ndim:
                        pids_mask_obj = pids_mask_obj.reshape(pids_mask_obj.shape[:-1])
                    datums['rgb'].append(rgb_obj)
                    datums['pids_mask'].append(pids_mask_obj)
                    datums['alpha'].append(mask_obj)
                    datums['bg_color'].append(bg_color_obj)
                    datums['rays_o'].append(rays_o_obj)
                    datums['rays_d'].append(rays_d_obj)
                    datums['coords'].append(coords_obj)
                
            for key in img_keys:
                # print(key)
                if not datums[key]:
                    datums[key] = np.zeros((0), dtype=np.float32)
                else:
                    datums[key] = np.concatenate(datums[key], axis=0)
                
            for name, obj in objs.items():
                # extract smpl params and keypoints3d for each human
                datums[name] = {key: obj[key] for key in obj_keys}
                # init near and far
                datums[name]['near'] = np.zeros(datums['rays_o'].shape[:-1], dtype=np.float32)
                datums[name]['far'] = np.ones(datums['rays_o'].shape[:-1], dtype=np.float32)
        # for validation and test
        else:
            datums.update({
                'rgb': img,
                'pids_mask': pids_mask,
                'alpha': pids_mask > 0,
                'bg_color': bg_color,
                'rays_o': rays_o,
                'rays_d': rays_d,
                'coords': self.coords,
            })
            
            for name, obj in objs.items():
                datum = {
                    'near': np.zeros(rays_o.shape[:-1], dtype=np.float32),
                    'far': np.ones(rays_o.shape[:-1], dtype=np.float32),
                    **{key: obj[key] for key in obj_keys}
                }
                datums[name] = datum
        return datums
    
    def get_img(self, info):
        imgname = info['imgname']
        cam_info = info['cam_info']
        cam = info['cam']
         
        # read image
        if self.split != 'test':
            img = cv2.imread(imgname)
        if self.split == 'test':
            img = np.zeros((self.img_size[0], self.img_size[1], 3)).astype('uint8')
        img, scale = scale_and_undistort(img, cam_info, cam, undistort=self.image_args.undis, res=self.img_size)
        return img, scale
    
    def update_info(self, info, scale):
        K = info['cam_info']['K']
        # update K
        K_new = K.copy()
        K_new[:2, -1] = K[:2, -1] * scale
        K_new[0, 0] = K[0, 0] * scale[0]
        K_new[1, 1] = K[1, 1] * scale[1]
        info['cam_info']['K'] = K_new
        info['cam_info']['invK'] = np.linalg.inv(K_new)
        if 'RT' in info['cam_info']:
            info['cam_info']['P'] = K_new @ info['cam_info']['RT']
        return info
    
    def get_coords(self):
        H, W = self.img_size
        img_ones = np.ones(self.img_size, dtype=np.float32)
        coords = np.argwhere(img_ones>0)
        coords = coords.reshape(H, W, 2)
        return coords
    
    def get_camnames(self, camnames):
        assert len(camnames) > 0
        camints = [int(cam) for cam in camnames]
        camints = sorted(camints)
        camnames = [str(cam) for cam in camints]
        return camnames
    
    def get_infos(self, root, cams_info, ranges, image_args):
        '''
        including camera, img informations
        (cam, frame) -> imgname
        '''
        infos = []
        # index = 0
        ranges = np.array(ranges).reshape(-1, 3)
        for r in range(len(ranges)):
            for nnf, nf in enumerate(range(*ranges[r])):
                for idx, cam in enumerate(self.camnames):
                    cam_info = cams_info[cam].copy()
                    imgname = join(root, image_args.root, cam, '{:06d}{}'.format(nf, image_args['ext']))
                    info = {
                        'cam': cam,
                        'cam_idx': idx,
                        'cam_info': cam_info,
                        'frame': nf,
                        'frame_idx': nnf,
                        'time_idx': nf - self.start_frame,
                        'imgname': imgname
                    }
                    infos.append(info)
        return infos
    
    def create_demo_cameras(self, scale, camera_args, demo_args=None):
        if camera_args.method == 'none':
            from AvatarPose.utils.utils_sampling import create_center_radius
            RTs = create_center_radius(**camera_args)
            K = np.array([
                camera_args.focal, 0, camera_args.W/2,
                0, camera_args.focal, camera_args.H/2,
                0, 0, 1], dtype=np.float32).reshape(3, 3)[None].repeat(RTs.shape[0], 0)
            R = RTs[:, :3, :3]
            T= RTs[:, :3, 3:]
        elif camera_args.method == 'mean':
            from AvatarPose.utils.utils_sampling import create_cameras_mean
            K, R, T = create_cameras_mean(list(self.cams_info.values()), camera_args)
            K[:, 0, 2] = camera_args.W / 2
            K[:, 1, 2] = camera_args.H / 2
        # elif camera_args.method == 'static':
        #     assert len(self.subs) == 1, "Only support monocular videos"
        #     camera = self.cameras[self.subs[0]]
        #     K = camera['K'][None]
        #     R = camera['R'][None]
        #     T = camera['T'][None]
        # elif camera_args.method == 'line':
        #     for key, camera in self.cameras.items():
        #         R = camera['R']
        #         T = camera['T']
        #         center_old = - R.T @ T
        #         print(key, center_old.T[0])
        #     camera = self.cameras[str(camera_args.ref_sub)]
        #     K = camera['K'][None]
        #     R = camera['R'][None]
        #     T = camera['T'][None]
        #     t = np.linspace(0., 1., camera_args.allstep).reshape(-1, 1)
        #     t = t - 0.33
        #     t[t<0.] = 0.
        #     t = t/t.max()
        #     start = np.array(camera_args.center_start).reshape(1, 3)
        #     end = np.array(camera_args.center_end).reshape(1, 3)
        #     center = end * t + start * (1-t)
        #     K = K.repeat(camera_args.allstep, 0)
        #     R = R.repeat(camera_args.allstep, 0)
        #     T = - np.einsum('fab,fb->fa', R, center)
        #     T = T.reshape(-1, 3, 1)
        K[:, :2] *= scale
        # create scripts
        if demo_args.mode == 'scripts':
            infos = self._demo_script(self.ranges, K, R, T, demo_args.stages)
        else:
            raise NotImplementedError
        return infos
    
    def _demo_script(self, ranges, K, R, T, stages):
        infos = []
        index = 0
        frames = [i for i in range(*ranges)]
        for name, stage in stages.items():
            _infos = []
            _frames = list(range(*stage.frame))
            _views = list(range(*stage.view))
            if len(_frames) == 1 and len(_views) != 1:
                _frames = _frames * len(_views)
            elif len(_views) == 1 and len(_frames) != 1:
                _views = _views * len(_frames)
            elif len(_views) == 1 and len(_frames) == 1 and 'steps' in stage.keys():
                _views = _views * stage.steps
                _frames = _frames * stage.steps
            elif len(_views) != 1 and len(_frames) != 1 and len(_views) != len(_frames):
                raise NotImplementedError
            _index = [i for i in range(len(_frames))]
            for _i in _index:
                nv, nf = _views[_i], _frames[_i]
                nv = nv % (K.shape[0])
                info = {
                    'imgname': 'none',
                    'cam': 'novel_'+str(nv),
                    'frame': nf,
                    'frame_idx': frames.index(nf),
                    'time_idx': nf - self.start_frame,
                    'cam_idx': nv,
                    'index': _i + index,
                    'cam_info': {
                        'K': K[nv],
                        'dist': np.zeros((1, 5)),
                        'R': R[nv],
                        'T': T[nv]
                    }
                }
                # create object
                float_i = _i*1./(len(_index) - 1)
                object_keys = stage.object_keys.copy()
                assert (len(object_keys) != 0)
                if 'effect' in stage.keys():
                    if stage.effect in ['disappear', 'appear']:
                        for _obj in stage.effect_args.key:
                            object_keys.remove(_obj)
                            if stage.effect == 'disappear':
                                occ = (1 - float_i)**3
                            elif stage.effect == 'appear':
                                occ = float_i**3
                            object_keys.append(_obj+"_@{{'scale_occ': {}, 'min_acc': 0.5}}".format(occ))
                    if stage.effect in ['zoom']:
                        scale = float_i * stage.effect_args.scale[1] + (1-float_i) * stage.effect_args.scale[0]
                        cx = float_i * stage.effect_args.cx[1] + (1-float_i) * stage.effect_args.cx[0]
                        cy = float_i * stage.effect_args.cy[1] + (1-float_i) * stage.effect_args.cy[0]
                        _K = info['camera']['K'].copy()
                        _K[:2, :2] *= scale
                        _K[0, 2] *= cx
                        _K[1, 2] *= cy
                        info['camera']['K'] = _K
                        info['camera']['K'] = _K
                        info['sub'] = info['sub'] + '_scale_{}'.format(scale)
                    if stage.effect_args.get('use_previous_K', False):
                        info['camera']['K'] = infos[-1]['camera']['K']
                info['object_keys'] = object_keys
                _infos.append(info)
            index += len(_index)
            infos.extend(_infos)
        return infos
    
    def vis_everything(self, ret, img_orig, objs, img, pids_mask):
        
        H = ret['meta']['H']
        W = ret['meta']['W']
        cam = ret['meta']['cam']
        frame_idx = ret['meta']['frame_idx']
        filedir = 'debug/vis_{}_{}'.format(cam, frame_idx)
        os.makedirs(filedir, exist_ok=True)
        
        # img orig
        img_orig = (np.clip(img_orig, 0, 1.) * 255).astype(np.uint8)
        img_orig = img_orig[..., [2, 1, 0]]
        cv2.imwrite(os.path.join(filedir, 'img_orig.jpg'), img_orig)
        
        # img mask
        mask = ((pids_mask>0)).astype(np.uint8)
        mask_3 = np.stack([mask*28, mask*163, mask*255], axis=-1)
        cv2.imwrite(os.path.join(filedir, 'mask.jpg'), mask_3)
        
        # img masked
        img_masked = (np.clip(img, 0, 1.) * 255).astype(np.uint8)
        img_masked = img_masked[..., [2, 1, 0]]
        cv2.imwrite(os.path.join(filedir, 'img_masked.jpg'), img_masked)
        
        # img sampled
        res = np.zeros((H, W, 3))
        coord = ret['coords'].reshape(-1, 2)
        rgb = ret['rgb'].reshape(-1, 3)
        res[coord[:, 0], coord[:, 1]] = rgb
        img_sampled = (np.clip(res, 0, 1.) * 255).astype(np.uint8)
        img_sampled = img_sampled[..., [2, 1, 0]]
        cv2.imwrite(os.path.join(filedir, 'img_sampled.jpg'), img_sampled)
        
        # # smpl mesh
        # from AvatarPose.vis.vis_mesh import VisMesh
        # from instant_avatar.deformers.smpl.smpl import SMPLServer
        # vertices = {}
        # pids = []
        # for name in ret.keys():
        #     if 'human' in name:
        #         smpl_obj = SMPLServer(name, betas = ret[name]['betas'])
        #         smpl_output = smpl_obj.posed_smpl(ret[name]['body_pose'], ret[name]['global_orient'], ret[name]['transl'])
        #         vertices[name] = smpl_output['smpl_verts'][0].numpy()
        #         pids.append(name)
        # vis_mesh = VisMesh(pids=pids, vertices=vertices)
        # vis_mesh.savemesh(filename=os.path.join(filedir,'meshes.ply'))

        # for name in pids:
        #     # vert projection
        #     vis_mesh.projection(ret['cam_info'], ret['meta']['H'], ret['meta']['W'], name, filename=os.path.join(filedir,'vert_proj_{}.jpg'.format(name)))
        #     # mesh projection
        #     vis_mesh.projection_mask(ret['cam_info'], ret['meta']['H'], ret['meta']['W'], name, filename=os.path.join(filedir,'mesh_proj_mask_{}.jpg'.format(name)))
       

 
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, opt, split_opt, **kwargs):
        super().__init__()
        data_dir = Path(hydra.utils.to_absolute_path(opt.root))
        for split in split_opt.keys():
            dataset = BaseDataset(data_dir, split, opt, split_opt.get(split))
            setattr(self, f"{split}set", dataset)
        self.opt = opt
        self.split_opt = split_opt

    def train_avatar_dataloader(self):
        if hasattr(self, "train_avatarset"):
            return DataLoader(self.train_avatarset,
                              shuffle=True,
                              num_workers=self.opt.num_workers,
                              persistent_workers=True and self.opt.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()
    
    def train_avatar_refine_dataloader(self):
        if hasattr(self, "train_avatar_refineset"):
            return DataLoader(self.train_avatar_refineset,
                              shuffle=True,
                              num_workers=self.opt.num_workers,
                              persistent_workers=True and self.opt.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()
        
    def train_smpl_dataloader(self):
        if hasattr(self, "train_smplset"):
            return DataLoader(self.train_smplset,
                              shuffle=True,
                              num_workers=self.opt.num_workers,
                              persistent_workers=True and self.opt.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()
    
    def train_arm_dataloader(self):
        if hasattr(self, "train_armset"):
            return DataLoader(self.train_armset,
                              shuffle=True,
                              num_workers=self.opt.num_workers,
                              persistent_workers=True and self.opt.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(self.valset,
                              shuffle=False,
                              num_workers=self.opt.num_workers,
                              persistent_workers=True and self.opt.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(self.testset,
                              shuffle=False,
                              num_workers=self.opt.num_workers,
                              persistent_workers=True and self.opt.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()