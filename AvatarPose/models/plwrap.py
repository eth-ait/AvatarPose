# Training code based on PyTorch-Lightning
import os
from os.path import join
import torch
import hydra
import numpy as np
import time
import pytorch_lightning as pl
from AvatarPose.utils.loss import Evaluator_Avatar, Evaluator_Pose
from AvatarPose.utils.utils import to_tensor, to_device, to_np
from AvatarPose.utils.file_utils import write_keypoints3d
from AvatarPose.utils.utils import get_gpu_memory
from AvatarPose.models.plbase import plbase


# https://github.com/Project-MONAI/MONAI/issues/701
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import logging
logger = logging.getLogger("mp-instant.plwrapper")
logger.addHandler(logging.FileHandler("mpinstant.log"))

class plwrapper(plbase):
    def __init__(self, opt, datamodule, mode='train'):
        super().__init__(opt, datamodule, mode)
        # load model
    
    def train_dataloader(self):
        self.stage, self.idx_range = self.find_epoch(self.current_epoch)
        splits = self.stage.split('_')
        for split in self.load_order:
            if split in splits:
                return self.dataloader_dict[split]
        raise ValueError(f'opt.stages contains {self.stage}: Stage not valid') 
        
    def on_train_epoch_start(self):
        self.stage, self.idx_range = self.find_epoch(self.current_epoch)
        if self.training:
            self.deactivate_grad_all()
            self.activate_grad([self.grad_dict[s] for s in self.stage.split('_')])
            scheduler = self.lr_schedulers()
            scheduler.step()  
        
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers(False)
        optimizer.zero_grad()
        
        if self.opt.joint_opt:
            losses = self.train_joint(batch)
        elif 'avatar' in self.stage or 'arefine' in self.stage:
            losses = self.train_avatar(batch)     
        else:
            losses = self.train_smpl(batch)
        # self.check_model_params()
        if losses is None:
            return None
        loss = losses["loss"]
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()       
        return None  
    
    def train_avatar(self, batch):
        if batch['meta']['names'] == []:
            return None
        self.opt_avatar=True
        names_all = [name[0] for name in batch['meta']['names']]
        assert self.training
        reg, pred_all = self.render_image(batch, names_all)
        losses = self.loss_avatar(batch, pred_all)
        losses_new = self.compute_loss(batch, pred_all, with_psnr=True)
        losses.update(losses_new)
        if reg is not None:
            losses["reg"] = reg
            losses["loss"] += reg
        log_name = f'{self.opt.vis_split}_{self.opt.vis_log_name}'
        self.set_log(log_name, losses, with_precision=True, with_grid=True, names_all=names_all)     
        return losses

    def train_smpl(self, batch):  
        assert self.training
        self.opt_avatar=False
        with_smpl_loss = self.opt.tracking
        pred_all, names_all = self.pred(batch, with_cano=False, with_kpts=True, with_smpl=True, with_smpl_loss=with_smpl_loss, with_smpl_density=True)
        losses = self.loss_smpl(batch, pred_all)
        losses_new = self.compute_loss(batch, pred_all, with_psnr=True)
        losses.update(losses_new)
        log_name = f'{self.opt.vis_split}_{self.opt.vis_log_name}'
        self.set_log(log_name, losses)
        return losses
    
    def train_joint(self, batch):
        assert self.training
        self.opt_avatar=True
        with_smpl_loss = self.opt.tracking
        pred_all, names_all = self.pred(batch, with_cano=False, with_kpts=True, with_smpl=True, with_smpl_loss=with_smpl_loss, with_smpl_density=True)
        losses = self.loss_smpl(batch, pred_all)
        losses_new = self.compute_loss(batch, pred_all, with_psnr=True)
        losses.update(losses_new)
        log_name = f'{self.opt.vis_split}_{self.opt.vis_log_name}'
        self.set_log(log_name, losses)
        return losses
    

    def render_image(self, batch, names_all, is_cano=False):
        # init instantavatar model
        reg_list = []
        for name in names_all:
            model = self.networks.get_model(name)
            eval_mode = not(self.opt_avatar and self.training)
            r = model.initialize(batch, self.global_step, eval_mode, self.opt, is_cano)  
            if r is not None:
                reg_list.append(r)
        reg = None
        if len(reg_list) != 0:
            reg = sum(reg_list)
            
        # predict rgb, alpha, depth, weight
        use_noise = self.global_step < 1000 and self.training and self.opt_avatar
        eval_mode = not(self.opt_avatar and self.training)
        pred_all = self.renderer(batch, names_all, use_noise=use_noise, eval_mode=eval_mode, layered=True)
        pred_all['step'] = self.global_step
        pred_all['epoch'] = self.current_epoch
        pred_all['names_all'] = names_all
        pred_all['time_idx'] = batch['meta']['time_idx']
        return reg, pred_all
    
    def pred(self, batch, with_cano=False, with_kpts=False, with_kpts_gt=False, with_smpl=False, with_smpl_loss=False, with_smpl_density=False):
        if not batch['meta']['names']:
            return None
        names_all = [name[0] for name in batch['meta']['names']]
        pred_all = {name:{} for name in names_all}
        
        # canonical pose
        if with_cano:
            for name in names_all:
                model = self.networks.get_model(name)
                datas_cano = model.preprocess_canonical(batch)
                _, pred_cano = self.render_image(datas_cano, [name], is_cano=True)
                pred_all[name]['rgb_cano'] = pred_cano['rgb']
                
        _, pred_all_new = self.render_image(batch, names_all)
        pred_all.update(pred_all_new)
        
        # keypoints3d
        if with_kpts:
            for name in names_all:
                model = self.networks.get_model(name)
                pred_all[name]['kpts'] = model.kpts #(1, n_joints, 3)
                
        # smpl params
        if with_smpl:
            for name in names_all:
                model = self.networks.get_model(name)
                smpl_params = model.SMPL_params(batch['meta']['time_idx'])
                for k in ['betas', 'body_pose', 'global_orient', 'transl']:
                    pred_all[name][k] = smpl_params[k]
                    
        if with_smpl_loss:
            for name in names_all:
                model = self.networks.get_model(name)
                loss_v, loss_a = model.SMPL_params.tv_loss(batch['meta']['time_idx'])
                pred_all[name]['loss_v'] = loss_v
                pred_all[name]['loss_a'] = loss_a
                
        if with_kpts_gt:
            time_idx = batch['meta']['time_idx'][0]
            device = time_idx.device
            kpts_gt = to_device(self.datamodule.valset.retrieve_kpts_gt(time_idx), device)
            for name in names_all:
                pred_all[name]['kpts_gt'] = kpts_gt[name]['kpts'].unsqueeze(0)
        # if with_smpl_density:
        #     opacity_coll = []
        #     for name in names_all:
        #         model = self.networks.get_model(name)
        #         joints = model.deformer.smpl_joints.reshape(-1, 3)
        #         for name_temp in names_all:
        #             if name_temp == name:
        #                 continue
        #             model_temp = self.networks.get_model(name_temp)
        #             joints_temp = model_temp.deformer.transform_points_w2s(joints)
        #             density = model_temp.cal_density(joints_temp)
        #             opacity = 1.0 - torch.exp(-torch.relu(density) * 0.01)
        #             opacity_coll.append(opacity)
        #     if opacity_coll == []:
        #         pred_all['smpl_joints_opacity'] = opacity_coll
        #     else:
        #         pred_all['smpl_joints_opacity'] = torch.cat(opacity_coll)
                    
        return pred_all, names_all
    
    def compute_loss(self, batch, pred_all, with_psnr=False, with_rgb=False, with_masked_rgb=False, with_kpts=False, with_smpl=False):
        losses = {}
        if with_psnr:
            losses['psnr'] = self.evaluator.psnr(batch['rgb'], pred_all['rgb'])
        if with_rgb:
            losses = {
                **self.evaluator(batch['rgb'], pred_all['rgb']),
                "rgb_loss": (batch['rgb'] - pred_all['rgb']).square().mean(),
            }
        if with_masked_rgb:
            losses['masked_rgb_loss'] = (batch['rgb'][batch['pids_mask']] - pred_all['rgb'][batch['pids_mask']]).square().mean()
        if with_kpts:
            time_idx = batch['meta']['time_idx'][0]
            device = time_idx.device
            kpts_gt = to_device(self.datamodule.valset.retrieve_kpts_gt(time_idx), device)
            kpts_init = to_device(self.datamodule.valset.retrieve_kpts_init(time_idx), device)
            losses['mpjpe_init'] = self.pose_evaluator.forward_kpts(kpts_init, pred_all)
            losses['mpjpe_gt'] = self.pose_evaluator.forward_kpts(kpts_gt, pred_all)
        if with_smpl:
            time_idx = batch['meta']['time_idx'][0]
            device = time_idx.device
            smpl_gt = to_device(self.datamodule.valset.retrieve_smpl_gt(time_idx), device)
            losses_smpl = self.pose_evaluator.forward_smpl(smpl_gt, pred_all)
            losses.update(losses_smpl)
        return losses
    
    def set_log(self, split, losses, with_precision=False, with_grid=False, names_all=None):
        for k, v in losses.items():
            self.log(f"{split}/{k}", v)
        _, used_m = get_gpu_memory()
        self.log(f'{split}/used_memory', used_m) 
        if with_precision:
            if self.precision == 16:
                self.log("precision/scale",
                     self.trainer.precision_plugin.scaler.get_scale())
        if with_grid:
            for name in names_all:
                model = self.networks.get_model(name)
                # log density grid field
                self.log(f"train/model_density_field_{name}", model.renderer.density_grid_train.density_field.sum())
            


        
    def save_kpts_smpl(self, pred_all, frame):
        epoch = self.current_epoch
        step = self.global_step
        frame = frame.detach().cpu().numpy()[0]
        smpl_all = {}
        for name in pred_all['names_all']:
            smpl_all[name] = {}
            for k in ['betas', 'body_pose', 'global_orient', 'transl', 'frames']:
                smpl_all[name][k] = []
        kpts = []
        for name in pred_all['names_all']:
            kpts.append({
                    'id': name.split('_')[-1],
                    'keypoints3d': to_np(pred_all[name]['kpts'].reshape(-1, 3)),
                })
            for k in ['betas', 'body_pose', 'global_orient', 'transl']:
                smpl_all[name][k].append(pred_all[name][k].detach().cpu().numpy())
            smpl_all[name]['frames'].append(frame)
        os.makedirs(f'kpts_{frame}', exist_ok=True)
        write_keypoints3d(join(f'kpts_{frame}', f'kpts_{epoch:03d}_{step:06d}.json'), kpts)
        os.makedirs(f'smpl_{frame}/smpl_{epoch:03d}_{step:06d}', exist_ok=True)
        np.savez(join(f'smpl_{frame}/smpl_{epoch:03d}_{step:06d}', 'smpl_all.npz'), **smpl_all)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        pred_all, names_all = self.pred(batch, with_cano=True, with_kpts=True, with_kpts_gt=True, with_smpl=True)
        losses = self.compute_loss(batch, pred_all, with_rgb=True, with_kpts=True, with_smpl=True)
        log_name = f'{self.opt.vis_split}_{self.opt.vis_log_name}'
        self.set_log(log_name, losses)
        # self.save_kpts_smpl(pred_all, batch['meta']['frame'])
        self.visualizer.set_names(log_name, f'{self.current_epoch:03d}_{self.global_step:06d}', batch['meta']['cam'][0], batch['meta']['frame'][0], self.global_step)
        if self.opt.vis_log_name in ['kpts_init', 'kpts_gt']:
            self.visualizer.set_results(self.opt.pose_vis_results)
        else:
            self.visualizer.set_results(['rgb', 'rgb_cano', 'instance', 'kpts'])
            # self.visualizer.set_results(['rgb', 'instance', 'kpts'])
        self.visualizer.saveimg(pred_all, batch)
        return None
    

    @torch.no_grad()
    def test_step(self, batch, batch_idx, *args, **kwargs):
        start = time.time()
        pred_all, names_all= self.pred(batch, with_cano=True, with_kpts=False)
        end = time.time()
        print('time', start-end)
        # visualize
        log_name = f'{self.opt.vis_split}_{self.opt.vis_log_name}'
        self.visualizer.set_names(log_name, self.current_epoch, batch['meta']['cam'][0], batch['meta']['frame'][0], 0)
        self.visualizer.set_results(['rgb', 'rgb_cano', 'instance', 'alpha', 'depth'])
        self.visualizer.saveimg(pred_all, batch)
