import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.cuda.amp import custom_fwd
import hydra
import numpy as np

OFFSET = 0.313262

class LossRGB(nn.Module):
    def __init__(self, name, weight) -> None:
        super().__init__()
        self.name = name
        self.weight = weight
    
    def forward(self, targets, predicts):
        if self.name == 'huber':
            loss_rgb = F.huber_loss(predicts["rgb"], targets["rgb"], reduction="mean", delta=0.1)
        else:
            raise NotImplementedError
        return loss_rgb

class LossAlpha(nn.Module):
    def __init__(self, name, weight, opt) -> None:
        super().__init__()
        self.name = name
        self.weight = weight
        self.opt = opt
    
    def forward(self, targets, predicts):
        if self.name == 'mse':
            if predicts['step'] < self.opt.end_step:
                loss_alpha = F.mse_loss(predicts["alpha"], targets["alpha"])
            else:
                loss_alpha = torch.tensor(0.).to(predicts['alpha'].device)
        elif self.name == 'ce':
            loss_alpha = F.binary_cross_entropy_with_logits(predicts["alpha"], targets["alpha"])
        else:
            raise NotImplementedError
        return loss_alpha

class LossMaskedRGB(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(self, targets, predicts):
        loss_masked_rgb = 0
        if len(predicts["rgb"].shape) == 5:
            mask = targets["alpha"] > 0
            loss_masked_rgb = F.mse_loss(predicts["rgb"][mask], targets["rgb"][mask])
        return loss_masked_rgb
    
class LossLpips(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        for param in self.lpips.parameters(): param.requires_grad=False
        self.weight = weight
    
    def forward(self, targets, predicts):
        loss_lpips=0
        if len(predicts["rgb"].shape) == 5:
            loss_lpips = self.lpips(predicts["rgb"].flatten(0, 1).permute(0, 3, 1, 2).clip(max=1),
                                    targets["rgb"].flatten(0, 1).permute(0, 3, 1, 2))
        return loss_lpips
    
class LossInstance(nn.Module):
    def __init__(self, name, weight) -> None:
        super().__init__()
        self.name = name
        self.weight = weight
    
    def forward(self, targets, predicts):
        if self.name == 'mse':
            loss_instance = 0
            for name in predicts['names_all']:
                pid = int(name.split('_')[-1])
                mask = (targets['pids_mask'] == 0) | (targets['pids_mask'] == pid + 1)
                pids_mask = (targets['pids_mask'] == pid + 1) 
                pids_mask = pids_mask.float()
                loss_instance += F.mse_loss(predicts['instance'][..., pid][mask], pids_mask[mask])
            loss_instance = loss_instance/len(predicts['names_all'])
        else:
            raise NotImplementedError
        return loss_instance
    
    
class LossMaskVertice(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
    def forward(self, targets, predicts):
        mask = targets['alpha'] 
        cnt = 0
        for name in predicts['names_all']:
            verts_project = predicts[name]['verts_project_coords']
            verts_project = verts_project[:, 1].reshape(1, -1)
            cnt += torch.sum((verts_project - mask) > 0)
        
        loss_maskvertice = cnt / verts_project.sum()
        
        return loss_maskvertice

class LossKeypoints3d(nn.Module):
    def __init__(self, name, weight) -> None:
        super().__init__()
        self.name = name
        self.weight = weight
    def forward(self, targets, predicts):
        if self.name == 'mpjpe':
            loss_keypoints3d = 0
            for name in predicts['names_all']:
                loss_joints = 1e8
                for name in predicts['names_all']:
                    loss_temp = (torch.sqrt(((targets[name]['kpts_init'] - predicts[name]['kpts']) ** 2).sum(axis=-1)) * 1000).mean()
                    if loss_temp < loss_joints:
                        loss_joints = loss_temp 
                loss_keypoints3d += loss_joints
            loss_keypoints3d = loss_keypoints3d/len(predicts['names_all'])
        else:
            raise NotImplementedError
        return loss_keypoints3d

class RegDepth(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(self, predicts):
        reg_depth = 0
        if len(predicts["rgb"].shape) == 5:
            instances = predicts["instance"].clone()
            for pid in range(instances.shape[-1]):
                pids_mask = instances[..., pid] > 0.5
                alpha_coarse = predicts["alpha"].clone()
                depth_coarse = predicts["depth"].clone()
                alpha_coarse[~pids_mask] = 0
                depth_coarse[~pids_mask] = 0
                alpha_sum = alpha_coarse.sum()
                depth_avg = (depth_coarse * alpha_coarse).sum() / (alpha_sum + 1e-3)
                reg_depth = alpha_coarse * (depth_coarse - depth_avg[..., None, None]).abs()
                reg_depth = reg_depth.mean()
        return reg_depth
    
class RegPose(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
    def forward(self, predicts):
        reg_pose = 0
        cnt = 0
        for name in predicts['names_all']:
            reg_pose += torch.sum(predicts[name]['body_pose']**2)
            cnt += 1
        reg_pose /= cnt
        return reg_pose
    
class RegShape(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
    def forward(self, predicts):
        reg_shape = 0
        cnt = 0
        for name in predicts['names_all']:
            reg_shape += torch.sum(predicts[name]['betas']**2)
            cnt += 1
        reg_shape /= cnt
        return reg_shape

class RegInstance(nn.Module):
    def __init__(self, weight, opt) -> None:
        super().__init__()
        self.weight = weight
        self.opt = opt
    
    def forward(self, predicts):
        if predicts['epoch'] < self.opt.start_epoch or predicts['epoch'] > self.opt.end_epoch:
            reg_instance = torch.tensor(0.).to(predicts['instance'].device)
        else:
            reg_instance = (-predicts['instance'] * torch.log(torch.clamp(predicts['instance'], min=1e-5))).mean()

        return reg_instance
    
class RegPenetration(nn.Module):
    def __init__(self, weight, opt) -> None:
        super().__init__()
        self.weight = weight
        self.opt = opt
    
    def forward(self, predicts):
        if predicts['epoch'] < self.opt.start_epoch or predicts['epoch'] > self.opt.end_epoch:
            reg_instance = torch.tensor(0.).to(predicts['rgb'].device)
        else:
            if predicts['collision'] is None or len(predicts['collision']) == 0:
                reg_instance = torch.tensor(0.).to(predicts['rgb'].device)
            else:
                reg_instance = predicts['collision'].mean()

        return reg_instance
    
# class RegPenetration(nn.Module):
#     def __init__(self, weight, opt) -> None:
#         super().__init__()
#         self.weight = weight
#         self.opt = opt
    
#     def forward(self, predicts):
#         if predicts['epoch'] < self.opt.start_epoch or predicts['epoch'] > self.opt.end_epoch:
#             reg_instance = torch.tensor(0.).to(predicts['collision'].device)
#         else:
#             if predicts['collision'] is None or len(predicts['collision']) == 0:
#                 reg_instance = torch.tensor(0.).to(predicts['collision'].device)
#             else:
#                 reg_instance = predicts['collision'].mean()

#         return reg_instance
    
class RegSMPLPenetration(nn.Module):
    def __init__(self, weight, opt) -> None:
        super().__init__()
        self.weight = weight
        self.opt = opt
    
    def forward(self, predicts):
        if predicts['epoch'] < self.opt.start_epoch or predicts['epoch'] > self.opt.end_epoch:
            reg_instance = torch.tensor(0.).to(predicts['rgb'].device)
        else:
            if predicts['smpl_joints_opacity'] is None or len(predicts['smpl_joints_opacity']) == 0:
                reg_instance = torch.tensor(0.).to(predicts['rgb'].device)
            else:
                if self.opt.all_params:
                    reg_instance = predicts['smpl_joints_opacity'].mean()
                else:
                    reg_instance = (predicts['smpl_joints_opacity'][predicts['smpl_joints_opacity'] > 0]).mean()

        return reg_instance
    
class RegSMPLPenetration(nn.Module):
    def __init__(self, weight, opt) -> None:
        super().__init__()
        self.weight = weight
        self.opt = opt
    
    def forward(self, predicts):
        if predicts['epoch'] < self.opt.start_epoch or predicts['epoch'] > self.opt.end_epoch:
            reg_instance = torch.tensor(0.).to(predicts['rgb'].device)
        else:
            if predicts['smpl_joints_opacity'] is None or len(predicts['smpl_joints_opacity']) == 0:
                reg_instance = torch.tensor(0.).to(predicts['rgb'].device)
            else:
                reg_instance = predicts['smpl_joints_opacity'].mean()

        return reg_instance
    
class RegInstanceMask(nn.Module):
    def __init__(self, weight, opt) -> None:
        super().__init__()
        self.weight = weight
        self.opt = opt
    
    def forward(self, predicts):
        if predicts['epoch'] < self.opt.start_epoch or predicts['epoch'] >= self.opt.end_epoch:
            reg_instance = torch.tensor(0.).to(predicts['instance'].device)
        else:
            instance = predicts['instance'][predicts['instance'] >0]
            reg_instance = (-predicts['instance'] * torch.log(torch.clamp(predicts['instance'], min=1e-5))).mean()

        return reg_instance
    
class RegAlpha(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(self, predicts):
        reg_alpha   = (-torch.log(torch.exp(-predicts["alpha"]) + torch.exp(predicts["alpha"] - 1))).mean() + OFFSET
        return reg_alpha
    

class RegWeight(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(self, predicts):
        reg_density = (-torch.log(torch.exp(-predicts["weight"]) + torch.exp(predicts["weight"] - 1))).mean() + OFFSET
        return reg_density

class RegWeightMask(nn.Module):
    def __init__(self, weight, opt) -> None:
        super().__init__()
        self.weight = weight
        self.opt = opt
    
    def forward(self, predicts):
        if predicts['epoch'] < self.opt.end_epoch and predicts['epoch'] >= self.opt.start_epoch:
            w = predicts['weight'][predicts['weight'] >0]
            reg_density = (-torch.log(torch.exp(-w) + torch.exp(w - 1))).mean() + OFFSET
        else:
            reg_density = torch.tensor(0.).to(predicts['weight'].device)
        
        return reg_density
    
class RegSmooth(nn.Module):
    def __init__(self, weight, v_weight, a_weight, tracking) -> None:
        super().__init__()
        self.v_weight = v_weight
        self.a_weight = a_weight
        self.weight = weight
        self.tracking = tracking
    
    def forward(self, predicts):
        if self.tracking:
            reg_smooth = 0
            for name in predicts['names_all']:
                reg_smooth += predicts[name]['loss_v'] * self.v_weight
                reg_smooth += predicts[name]['loss_a'] * self.a_weight
            return reg_smooth
        else:
            return 0
        
class RegAngle(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight
        self.prior = SMPLifyAnglePrior()
    
    def forward(self, predicts):
        reg_angle = 0
        cnt = 0
        for name in predicts['names_all']:
            reg_angle += self.prior(predicts[name]['body_pose'])
            cnt += 1
        reg_angle /= cnt
        return torch.mean(reg_angle)
    


class SMPLifyAnglePrior(nn.Module):
    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()

        # Indices for the roration angle of
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        # self.register_buffer('angle_prior_idxs', angle_prior_idxs)
        self.angle_prior_idxs=angle_prior_idxs

        angle_prior_signs = np.array([1, -1, -1, -1],
                                     dtype=np.float32 if dtype == torch.float32
                                     else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs,
                                         dtype=dtype)
        # self.register_buffer('angle_prior_signs', angle_prior_signs)
        self.angle_prior_signs=angle_prior_signs

    def forward(self, pose, with_global_pose=False):
        ''' Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        '''
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] *
                         self.angle_prior_signs.to(pose.device)).pow(2)

    
class LossWrapper(nn.Module):
    def __init__(self, opt_loss, opt_reg):
        super().__init__()
        losses = {key: hydra.utils.instantiate(value) for key, value in opt_loss.items()}
        losses_reg = {key: hydra.utils.instantiate(value) for key, value in opt_reg.items()}
        self.loss_fns = nn.ModuleDict(losses)
        self.loss_reg_fns = nn.ModuleDict(losses_reg)
        
    def forward(self, target, pred):
        # calculate loss
        losses = {}
        loss = 0
        for key, func in self.loss_fns.items():
            
            val = func(target, pred)
            losses[key] = val
            loss += func.weight * val
        
        for key, func in self.loss_reg_fns.items():
            val = func(pred)
            losses[key] = val
            loss += func.weight * val           
        losses['loss'] = loss
        return losses
        
        
        

class Evaluator_Avatar(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }
    

class Evaluator_Pose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward_kpts(self, targets, predicts):
        loss_kpts = 0
        for name in predicts['names_all']:
            loss_joints = 1e8
            for name in predicts['names_all']:
                loss_temp = (torch.sqrt(((targets[name]['kpts'] - predicts[name]['kpts']) ** 2).sum(axis=-1)) * 1000).mean()
                if loss_temp < loss_joints:
                    loss_joints = loss_temp
            loss_kpts += loss_temp
        loss_kpts = loss_kpts/len(predicts['names_all'])
        
        return loss_kpts
    
    def forward_smpl(self, targets, predicts):
        loss_smpl = {'betas': 0, 'body_pose': 0, 'global_orient': 0, 'transl': 0}
        for k in loss_smpl.keys():
            for name in predicts['names_all']:
                loss_joints = 1e8
                for name in predicts['names_all']:
                    loss_temp = F.l1_loss(targets[name][k], predicts[name][k])
                    if loss_temp < loss_joints:
                        loss_joints = loss_temp
                loss_smpl[k] += loss_temp
            loss_smpl[k] = loss_smpl[k]/len(predicts['names_all'])
        
        return loss_smpl