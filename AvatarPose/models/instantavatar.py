import hydra
import torch
from torch import nn
from instant_avatar.deformers.smpl_deformer import SMPLDeformer
from instant_avatar.models.networks.ngp import NeRFNGPNet

class InstantAvatar(nn.Module):
    """
    InstantAvatar is a class that contains all the components of the InstantAvatar model.
    One InstantAvatar class represents one human.
    """
    def __init__(self, network, renderer, deformer, smpl, pid, datamodule):
        super().__init__()
        self.pid = pid
        self.name = f'human_{pid}'
        self.net_coarse = hydra.utils.instantiate(network)
        self.deformer = hydra.utils.instantiate(deformer)
        self.renderer = hydra.utils.instantiate(renderer)
        self.renderer.initialize(0)
        self.SMPL_params = hydra.utils.instantiate(smpl, **datamodule.train_smplset.retrieve_smpl(self.name))
        self.current = None
        
    def prepare_deformer(self, data):
        '''
        initalize the SMPL deformer with SMPL params
        '''
        self.deformer.prepare_deformer(data)
        self.kpts = self.deformer.kpts
    
    def update_density_grid(self, global_step, opt):
        N = 1 if opt.get("smpl_init", False) else 20
        resume = self.renderer.density_grid_train.density_field.sum() == 0
        if (global_step % N == 0 and hasattr(self.renderer, "density_grid_train")) or resume:
            density, valid = self.renderer.density_grid_train.update(self.deformer,
                                                                    self.net_coarse,
                                                                    global_step)
            reg = N * density[~valid].mean()
            if global_step < 500:
                reg += 0.5 * density.mean()
            return reg
        else:
            return None
        
    def forward(self, rays, bg_color, eval_mode=False, use_noise=False, layer=False):
        ret = self.renderer(rays,
                            lambda x, _: self.deformer(x, self.net_coarse, eval_mode),
                            eval_mode=eval_mode,
                            noise=1 if use_noise else 0,
                            bg_color=bg_color,
                            layer=layer)
        return ret
    
    def cal_density(self, pts):
        model = lambda x, _: self.deformer(x, self.net_coarse, True)
        _, sigma= model(pts, None)
        return sigma
    
    def prepare_smpl(self, batch, opt):
        '''
        input: batch with original smpl params
        output: batch with calculated smpl params
        '''
        if not opt.use_smpl_init:
            body_params = self.SMPL_params(batch['meta']['time_idx'])
            for k in ["global_orient", "body_pose", "transl"]:
                assert batch[self.name][k].shape == body_params[k].shape
                batch[self.name][k] = body_params[k]  
                
            # update betas if use SMPLDeformer
            if isinstance(self.deformer, SMPLDeformer):
                assert batch[self.name]['betas'].shape == body_params['betas'].shape
                batch[self.name]["betas"] = body_params["betas"]
                
            # update near & far with refined SMPL
            dist = torch.norm(batch[self.name]["transl"], dim=-1, keepdim=True).detach()
            batch[self.name]["near"] = dist - 1
            batch[self.name]["far"] = dist + 1
        return batch
    
    def preprocess_canonical(self, batch):
        '''
        input: original batch input
        output: batch input for canonical space
        '''
        datas_cano = {'meta':{'time_idx':batch['meta']['time_idx']}}
        datas_cano[self.name] = batch[self.name].copy()
        datas_cano[self.name]['body_pose'][:] = 0
        datas_cano[self.name]['body_pose'][:, 2] = 0.1
        datas_cano[self.name]['body_pose'][:, 5] = -0.1
        datas_cano[self.name]['body_pose'][:, 47] = -0.95
        datas_cano[self.name]['body_pose'][:, 50] = 0.95
        for key in ['bg_color', 'rays_o', 'rays_d']:
            datas_cano[key] = batch[key].clone()
        return datas_cano
    
    def initialize(self, batch, global_step, eval_mode, opt, is_cano=False):
        # use SMPL paramter optimization
        if not is_cano:
            batch = self.prepare_smpl(batch, opt)
            
        # load deformer with smpl params
        self.prepare_deformer(batch[self.name])
        
        # initialize coarse network
        if isinstance(self.net_coarse, NeRFNGPNet):
            self.net_coarse.initialize(self.deformer.bbox, batch['meta']['time_idx'], self.name)
            
        # update density grid
        if not eval_mode:
            r = self.update_density_grid(global_step, opt)
            return r
        else:
            if hasattr(self.renderer, "density_grid_test"):
                self.renderer.density_grid_test.initialize(self.deformer, self.net_coarse)
            return None
        
        

    