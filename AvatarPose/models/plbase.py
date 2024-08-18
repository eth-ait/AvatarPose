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


# https://github.com/Project-MONAI/MONAI/issues/701
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import logging
logger = logging.getLogger("mp-instant.plwrapper")
logger.addHandler(logging.FileHandler("mpinstant.log"))

class plbase(pl.LightningModule):
    def __init__(self, opt, datamodule, mode='train'):
        super().__init__()
        # load model
        self.opt = opt
        self.networks = hydra.utils.instantiate(opt.networks, datamodule=datamodule, _recursive_=False)
        self.renderer = hydra.utils.instantiate(opt.rendererwrap, net=self.networks,  _recursive_=False)
        self.visualizer = hydra.utils.instantiate(opt.visualizer, _recursive_=False)
        self.loss_avatar = hydra.utils.instantiate(opt.loss_avatar, _recursive_=False)
        self.loss_smpl = hydra.utils.instantiate(opt.loss_smpl, _recursive_=False)
        self.automatic_optimization = False
        self.datamodule = datamodule
        self.evaluator = Evaluator_Avatar()
        self.pose_evaluator = Evaluator_Pose()
        self.opt_avatar=False
        self.stage=None
        self.idx_range = -1
        self.config_parameters()
        self.config_dataloaders()
        self.config_grad()
        
    def config_dataloaders(self):
        self.load_order = ['avatar', 'arefine', 'pose', 'transl', 'rot', 'arm']
        self.dataloader_dict = {}
        for split in self.datamodule.split_opt:
            if split == 'train_avatar':
                self.dataloader_dict['avatar'] = self.datamodule.train_avatar_dataloader()
            elif split == 'train_avatar_refine':
                self.dataloader_dict['arefine'] = self.datamodule.train_avatar_refine_dataloader()
            elif split == 'train_smpl':
                self.dataloader_dict['pose'] = self.datamodule.train_smpl_dataloader()
                self.dataloader_dict['transl'] = self.datamodule.train_smpl_dataloader()
                self.dataloader_dict['rot'] = self.datamodule.train_smpl_dataloader()
            elif split == 'train_arm':
                self.dataloader_dict['arm'] = self.datamodule.train_arm_dataloader()
                
                
        
        
    def config_grad(self):
        self.grad_dict = {
            'avatar': 'avatar',
            'arefine': 'avatar',
            'betas': 'betas',
            'transl': 'transl', 
            'rot': 'global_orient',
            'pose': 'body_pose',
            'arm': 'body_pose.1'
        }
    
    def config_parameters(self):
        if self.opt.pose_seperate:
            self.param_keys = ["delta_betas", "delta_transl", "delta_global_orient", "mlps.body_pose.0", "mlps.body_pose.1", "encoder"]
        else:
            self.param_keys = ["delta_betas", "delta_transl", "delta_global_orient", "mlps.body_pose", "encoder"]
            
        self.lr_dict = {
            'delta_betas': self.opt.SMPL_lr.betas,
            'delta_transl': self.opt.SMPL_lr.transl,
            'delta_global_orient': self.opt.SMPL_lr.rot,
            'mlps.body_pose.0': self.opt.SMPL_lr.pose_0,
            'mlps.body_pose.1': self.opt.SMPL_lr.pose_1,
            'mlps.body_pose': self.opt.SMPL_lr.pose,
        }
    
    def configure_optimizers(self):
        print('configure_optimizers')
       
        # create parameters dict
        params_dict = {}  
        for key in self.param_keys:
            params_dict[key] = {'names': [], 'params': []}
        params_dict['other'] = {'names': [], 'params': []}
        for (name, param) in self.named_parameters():
            if name.startswith("evaluator"):
                continue
            flag = 0
            for key in self.param_keys:
                if key in name:
                    params_dict[key]['names'].append(name)
                    params_dict[key]['params'].append(param)
                    flag = 1
            if not flag:
                params_dict['other']['names'].append(name)
                params_dict['other']['params'].append(param)
                
        if self.opt.pose_seperate:
            assert all('mlps.body_pose' not in name for name in params_dict['other']['names']), 'check the value of opt.pose_seperate or check the name of parameters related to SMPL body_pose'
        
        # create optimizer
        params_lr = []
        for name in params_dict.keys():
            if self.lr_dict.get(name):
                params_lr.append({'params': params_dict[name]['params'], 'lr': self.lr_dict[name]})
            else:
                params_lr.append({'params': params_dict[name]['params']})
        optimizer = torch.optim.Adam(params_lr, **self.opt.optimizer)
                         
        def lr_lambda(epoch):
            if self.stage is None:
                return 0
            ranges = self.opt.stages[self.stage][self.idx_range]
            index = self.current_epoch - ranges[0]
            return (1 - index / (ranges[1] - ranges[0])) ** 1.5
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda)
        # additional configure for gradscaler
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)
        return [optimizer], [scheduler]
    
    def load_avatar_weights(self, state_dict):
        model_dict = self.state_dict()
        avatar_dict = {k:v for k, v in state_dict.items() if 'net_coarse' in k}
        model_dict.update(avatar_dict)
        self.load_state_dict(model_dict)
    
    def load_smpl_weights(self, state_dict):
        for key in self.networks.state_dict().keys():
            if 'SMPL_params' in key:
                key_new = f'networks.{key}'
                self.networks.state_dict()[key] = state_dict[key_new]
    
    
    def find_epoch(self, epoch):
        for key in self.opt.stages.keys():
            for i in range(len(self.opt.stages[key])):
                stage = self.opt.stages[key][i]
                if epoch<stage[1] and epoch >=stage[0]:
                    return key, i
        return None, None
   
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()
    
    def test_dataloader(self):
        return self.datamodule.test_dataloader()
        
    
    
    def deactivate_grad_all(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            param.grad = None
    def deactivate_grad_SMPL(self):
        for name, param in self.named_parameters():
            if ('SMPL_params' in name):
                param.requires_grad = False   
                param.grad = None 
                         
    def activate_grad(self, keys=[]):
        if keys == []: 
            Warning('No keys to activate grad')
        for name, param in self.named_parameters():
            for key in keys:
                if key in ['avatar', 'arefine']:
                    if 'SMPL_params' not in name and 'lpips' not in name:
                        param.requires_grad = True
                elif key in name:
                    param.requires_grad = True
                
    def activate_grad_avatar(self):
        for name, param in self.named_parameters():
            if 'SMPL_params' not in name and 'lpips' not in name:
                param.requires_grad = True
    
    def deactivate_grad_avatar(self):
        for name, param in self.named_parameters():
            if 'SMPL_params' not in name and 'lpips' not in name:
                param.requires_grad = False
                param.grad = None
                
    def deactivate_grad_lpips(self):
        for name, param in self.named_parameters():
            if 'lpips' in name:
                param.requires_grad = False
                param.grad = None
    
    def check_model_params(self, ignore='lpips'):
        for name, param in self.named_parameters():
            if ignore in name:
                assert param.requires_grad == False
                continue
            print(name, param.requires_grad)
