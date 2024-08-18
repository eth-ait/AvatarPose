import torch
import torch.nn as nn

class SMPLParamRegression(nn.Module):
    """optimize SMPL params on the fly"""
    def __init__(self, vis_log_name, use_smpl_init, with_time_embed=False, pose_seperate=False,
                 **kwargs) -> None:
        super().__init__()
        assert('times' in kwargs.keys()), 'SMPL params must have detected times'
        # fill in init value
        self.vis_log_name = vis_log_name
        self.use_smpl_init = use_smpl_init
        self.pose_seperate = pose_seperate
        self.pose_idxs = [torch.arange(0, 51, 1), torch.arange(51, 69, 1)]
        self.conf_idxs = [torch.arange(0, 18, 1), torch.arange(18, 24, 1)]
        self.hidden_dim = 128
        self.depth = 3
        self.init_val = 1e-5
        self.with_conf = False
        if 'conf' in kwargs.keys():
            self.conf = kwargs['conf']
            self.with_conf = True
        self.with_time_embed = with_time_embed
        self.embed_dim = 64
        for k, v in kwargs.items():
            if k == 'times':
                self.times = v
            elif k == "betas":
                if v.shape[1]>1 and (v[0] != v[1]).any():
                    ValueError("all betas should be the same")
                self.betas = v[0]
                self.delta_betas = nn.Embedding(*v[0].shape)
                self.delta_betas.weight.data.uniform_(-self.init_val, self.init_val)
            elif k == 'global_orient':
                self.global_orient = v
                delta = [nn.Embedding(*v[i].shape) for i in range(v.shape[0])]
                for emb in delta:
                    emb.weight.data.uniform_(-self.init_val, self.init_val)
                self.delta_global_orient = nn.ModuleList(delta)
                # self.delta_global_orient = nn.Parameter(torch.zeros_like(self.global_orient))
            elif k == 'transl':
                self.transl = v
                delta = [nn.Embedding(*v[i].shape) for i in range(v.shape[0])]
                for emb in delta:
                    emb.weight.data.uniform_(-self.init_val, self.init_val)
                self.delta_transl = nn.ModuleList(delta)
                # self.delta_transl = nn.Parameter(torch.zeros_like(self.transl))
            elif k == "body_pose":
                self.body_pose = v
                pose_dict = self.init_pose_mlp(v)
        self.mlps = nn.ModuleDict(pose_dict)
        if self.with_time_embed:
            self.time_latent = nn.Embedding(len(self.times), self.embed_dim) 
        self.keys = ["betas", "global_orient", "transl", "body_pose"]
    
    def init_pose_mlp(self, v):
        pose_dict = {}
        extra_dim = 0
        extra_dim_list = [0, 0]
        if self.with_conf:
            extra_dim += self.conf.shape[-1]
            extra_dim_list[0] += self.conf_idxs[0].shape[0]
            extra_dim_list[1] += self.conf_idxs[1].shape[0]
        if self.with_time_embed:
            extra_dim += self.embed_dim
            extra_dim_list[0] += self.embed_dim
            extra_dim_list[1] += self.embed_dim
        if self.pose_seperate:
            pose_dict['body_pose'] = {}
            for i in range(len(self.pose_idxs)):
                pose_dict['body_pose'][str(i)] = self.create_mlp(self.pose_idxs[i].shape[0]+extra_dim_list[i], self.pose_idxs[i].shape[0])
            pose_dict['body_pose'] = nn.ModuleDict(pose_dict['body_pose'])
        else:
            pose_dict['body_pose'] = self.create_mlp(v.shape[-1]+extra_dim, v.shape[-1])
        return pose_dict
    
    def create_mlp(self, in_dim, out_dim):
        block_mlps = [nn.Linear(in_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(0, self.depth-1):
            block_mlps += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        block_mlps += [nn.Linear(self.hidden_dim, out_dim)]
        model = nn.Sequential(*block_mlps)
        
        last_layer = model[-1]
        last_layer.weight.data.uniform_(-self.init_val, self.init_val)
        last_layer.bias.data.zero_()
        return model
    

    
    def cal_pose(self, pose, idx):
        if self.with_conf:
            pose = self.cal_pose_conf(pose, idx)
        else:
            pose = self.cal_pose_woconf(pose, idx)
        return pose
            
    def cal_pose_conf(self, pose, idx):
        if self.pose_seperate:
            for i in range(len(self.pose_idxs)):
                pose_i = pose[:, self.pose_idxs[i]]
                conf_i = self.conf[idx][:, self.conf_idxs[i]]
                pose_i = pose_i + self.mlps['body_pose'][str(i)](torch.cat([pose_i, conf_i], dim=1))
                pose[:, self.pose_idxs[i]] = pose_i
        else:
            conf = self.conf[idx]
            pose = pose + self.mlps['body_pose'](torch.cat([pose, conf], dim=1))
        return pose
    
    def cal_pose_woconf(self, pose, idx):
        idx = torch.tensor([idx], device=pose.device)
        if self.pose_seperate:
            for i in range(len(self.pose_idxs)):
                pose_i = pose[:, self.pose_idxs[i]]
                input=pose_i
                if self.with_time_embed:
                    input = torch.cat([pose_i, self.time_latent(idx)], dim=1)
                pose_i = pose_i + self.mlps['body_pose'][str(i)](input)
                pose[:, self.pose_idxs[i]] = pose_i
        else:
            input = pose
            if self.with_time_embed:
                input = torch.cat([pose, self.time_latent(idx)], dim=1)
            pose = pose + self.mlps['body_pose'](input)
        return pose
    
    def forward(self, time):
        # time: example: tensor([1], device='cuda:0')
        # idx: example: 1 (int)
        idx = self.times.index(time[0])
        # idx = torch.tensor([idx], device=time.device)
        smpl_params = {
        "betas": self.betas,
        "body_pose": self.body_pose[idx],
        "global_orient": self.global_orient[idx],
        "transl": self.transl[idx]
        }
        smpl_params = {key:val.to(time.device) for key, val in smpl_params.items()}
        return {
            "betas": smpl_params['betas'] + self.delta_betas(torch.zeros_like(time)),
            "body_pose": self.cal_pose(smpl_params['body_pose'], idx),
            "global_orient": smpl_params['global_orient'] + self.delta_global_orient[idx](torch.zeros_like(time)),
            "transl": smpl_params['transl'] + self.delta_transl[idx](torch.zeros_like(time)),
        }

        
    def grad_check(self):
        main_names = ['betas', 'global_orient', 'transl', 'body_pose', 'body_pose.0', 'body_pose.1']
        grad = [True] * len(main_names)
        for name, param in self.named_parameters():
            for idx, key in enumerate(main_names):
                if key in name:
                    grad[idx] = grad[idx] * param.requires_grad
        print(main_names, grad)
        
    def tv_loss(self, time):
        loss_v = 0
        loss_a = 0

        max_time = max(self.times)
        min_time = min(self.times)
        idx_p = (time - 1).clip(min=min_time)
        idx_n = (time + 1).clip(max=max_time)
        v = self.forward(time)
        v_p = self.forward(idx_p)
        v_n = self.forward(idx_n)
        for k in v.keys():
            if k == "betas":
                continue
            if k=='transl':
                # velocity
                loss_v += (v[k] - v_p[k]).abs().mean()
                loss_v += (v_n[k] - v[k]).abs().mean()
                # acceleration
                loss_a += (v_p[k] - 2*v[k] + v_n[k]).abs().mean()
            if k == 'globel_orient' or k == 'body_pose':
                # velocity
                ang = self.angle_rot(v[k])
                ang_p = self.angle_rot(v_p[k])
                ang_n = self.angle_rot(v_n[k])
                loss_v += (ang - ang_p).square().mean()
                loss_v += (ang_n - ang).square().mean()
                loss_v += (ang[:2] - ang_p[:2]).square().mean() * 10
                # acceleration
                loss_a += (ang_p - 2*ang + ang_n).square().mean()
        return loss_v, loss_a
    
    def angle_rot(self, pose):
        pose = pose.reshape(-1, 3)
        angle = torch.norm(pose, dim=1)
        return angle


class SMPLParamEmbedding_2(nn.Module):
    """optimize SMPL params on the fly"""
    def __init__(self, tracking=False, use_smpl_init=False,
                 **kwargs) -> None:
        super().__init__()
        assert('times' in kwargs.keys()), 'SMPL params must have detected times'
        # fill in init value
        self.use_smpl_init = use_smpl_init 
        self.tracking = tracking
        for k, v in kwargs.items():
            if k == "betas":
                setattr(self, k, nn.Embedding.from_pretrained(v[0], freeze=False))
            elif k == "times":
                self.times = v
            else:
                embeds = [nn.Embedding.from_pretrained(v[i], freeze=False) for i in range(v.shape[0])]
                setattr(self, k, nn.ModuleList(embeds))
        self.keys = ["betas", "global_orient", "transl", "body_pose"]

    def forward(self, time):
        # time: example: tensor([1], device='cuda:0')
        # idx: example: 1 (int)
        idx = self.times.index(time[0])
        if not self.use_smpl_init:
            if not self.tracking:
                for name, param in self.named_parameters():
                    # if 'SMPL_params' in name:
                    if (f'body_pose.{int(idx)}.weight' in name) or (f'betas.weight' in name) or (f'global_orient.{int(idx)}.weight' in name) or (f'transl.{int(idx)}.weight' in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for name, param in self.named_parameters():
                    if 'SMPL_params' in name:
                        param.requires_grad = True
        return {
            "betas": self.betas(torch.zeros_like(time)),
            "body_pose": self.body_pose[idx](torch.zeros_like(time)),
            "global_orient": self.global_orient[idx](torch.zeros_like(time)),
            "transl": self.transl[idx](torch.zeros_like(time)),
        }
        
    def tv_loss(self, time):
        loss_v = 0
        loss_a = 0

        max_time = max(self.times)
        min_time = min(self.times)
        idx_p = (time - 1).clip(min=min_time)
        idx_n = (time + 1).clip(max=max_time)
        v = self.forward(time)
        v_p = self.forward(idx_p)
        v_n = self.forward(idx_n)
        for k in v.keys():
            if k == "betas":
                continue
            # velocity
            loss_v += (v[k] - v_p[k]).square().mean()
            loss_v += (v_n[k] - v[k]).square().mean()
            # acceleration
            loss_a += (v_p[k] - 2*v[k] + v_n[k]).square().mean()
        return loss_v, loss_a
    

class SMPLParamEmbedding(nn.Module):
    """optimize SMPL params on the fly"""
    def __init__(self,
                 **kwargs) -> None:
        super().__init__()

        # fill in init value
        for k, v in kwargs.items():
            v = v.reshape(v.shape[0], v.shape[2])
            setattr(self, k, nn.Embedding.from_pretrained(v, freeze=False))
        self.keys = ["betas", "global_orient", "transl", "body_pose"]

    def forward(self, idx):
        return {
            "betas": self.betas(torch.zeros_like(idx)),
            "body_pose": self.body_pose(idx),
            "global_orient": self.global_orient(idx),
            "transl": self.transl(idx),
        }

    def tv_loss(self, idx):
        loss = 0

        N = len(self.global_orient.weight)
        idx_p = (idx - 1).clip(min=0)
        idx_n = (idx + 1).clip(max=N - 1)
        for (k, v) in self.items():
            if k == "betas":
                continue
            loss = loss + (v(idx) - v(idx_p)).square().mean()
            loss = loss + (v(idx_n) - v(idx)).square().mean()
        return loss