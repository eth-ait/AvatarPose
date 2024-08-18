import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class NGPLoss(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        for param in self.lpips.parameters(): param.requires_grad=False
        self.opt = opt

    def forward(self, predicts, targets):
        losses = {}
        loss = 0

        loss_rgb = F.huber_loss(predicts["rgb_coarse"], targets["rgb"], reduction="mean", delta=0.1)
        loss += self.opt.w_rgb * loss_rgb
        losses["huber_loss"] = loss_rgb

        loss_alpha = F.mse_loss(predicts["alpha_coarse"], targets["alpha"])
        # loss_alpha=0
        # loss += self.opt.w_alpha * loss_alpha
        losses["loss_alpha_coarse"] = loss_alpha

        if self.opt.get("w_lpips", 0) > 0 and len(predicts["rgb_coarse"].shape) == 5:
            loss_lpips = self.lpips(predicts["rgb_coarse"].flatten(0, 1).permute(0, 3, 1, 2).clip(max=1),
                                    targets["rgb"].flatten(0, 1).permute(0, 3, 1, 2))
            losses["loss_lpips"] = loss_lpips
            loss += loss_lpips * self.opt.w_lpips
        
        if self.opt.get("w_depth_reg", 0) > 0 and len(predicts["rgb_coarse"].shape) == 5:
            alpha_sum = predicts["alpha_coarse"].sum()
            depth_avg = (predicts["depth_coarse"] * predicts["alpha_coarse"]).sum() / (alpha_sum + 1e-3)
            loss_depth_reg = predicts["alpha_coarse"] * (predicts["depth_coarse"] - depth_avg[..., None, None]).abs()
            loss_depth_reg = loss_depth_reg.mean()
            losses["loss_depth_reg"] = loss_depth_reg
            loss += self.opt.w_depth_reg * loss_depth_reg

        OFFSET = 0.313262
        reg_alpha   = (-torch.log(torch.exp(-predicts["alpha_coarse"]) + torch.exp(predicts["alpha_coarse"] - 1))).mean() + OFFSET
        reg_density = (-torch.log(torch.exp(-predicts["weight_coarse"]) + torch.exp(predicts["weight_coarse"] - 1))).mean() + OFFSET
        losses["reg_alpha"] = reg_alpha
        losses["reg_density"] = reg_density
        loss += self.opt.w_reg * reg_alpha
        loss += self.opt.w_reg * reg_density

        losses["loss"] = loss
        return losses
    
    
class MPNGPLoss(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        for param in self.lpips.parameters(): param.requires_grad=False
        self.opt = opt

    def forward(self, predicts, targets):
        losses = {}
        loss = 0

        loss_rgb = F.huber_loss(predicts["rgb_coarse"], targets["rgb"], reduction="mean", delta=0.1)
        loss += self.opt.w_rgb * loss_rgb
        losses["huber_loss"] = loss_rgb

        if predicts['step'] < self.opt.end_mask:
            loss_alpha = F.mse_loss(predicts["alpha_coarse"], targets["alpha"])
            loss += self.opt.w_alpha * loss_alpha
            losses["loss_alpha_coarse"] = loss_alpha
        else:
            losses['loss_alpha_coarse'] = torch.tensor(0.).to(predicts['alpha_coarse'].device)
            loss += losses['loss_alpha_coarse']
            

        if self.opt.get("w_lpips", 0) > 0 and len(predicts["rgb_coarse"].shape) == 5:
            loss_lpips = self.lpips(predicts["rgb_coarse"].flatten(0, 1).permute(0, 3, 1, 2).clip(max=1),
                                    targets["rgb"].flatten(0, 1).permute(0, 3, 1, 2))
            losses["loss_lpips"] = loss_lpips
            loss += loss_lpips * self.opt.w_lpips
        
        
        if self.opt.get("w_depth_reg", 0) > 0 and len(predicts["rgb_coarse"].shape) == 5:
            for pid in targets['pids_sampled']:
                pids_mask = (targets['pids_mask'] == pid + 1)
                alpha_coarse = predicts["alpha_coarse"].clone()
                depth_coarse = predicts["depth_coarse"].clone()
                alpha_coarse[~pids_mask] = 0
                depth_coarse[~pids_mask] = 0
                alpha_sum = alpha_coarse.sum()
                depth_avg = (depth_coarse * alpha_coarse).sum() / (alpha_sum + 1e-3)
                loss_depth_reg = alpha_coarse * (depth_coarse - depth_avg[..., None, None]).abs()
                loss_depth_reg = loss_depth_reg.mean()
                losses["loss_depth_reg"] = loss_depth_reg
                loss += self.opt.w_depth_reg * loss_depth_reg
                
        # instance mask loss
        if predicts['step'] < self.opt.start_instance or predicts['step'] > self.opt.end_instance:
            losses['reg_instance'] = torch.tensor(0.).to(predicts['instance_coarse'].device)
            loss += losses['reg_instance']
        else:
            reg_instance = (-predicts['instance_coarse'] * torch.log(torch.clamp(predicts['instance_coarse'], min=1e-5))).mean()
            losses['reg_instance'] = reg_instance
            loss += self.opt.w_instance * reg_instance

        OFFSET = 0.313262
        reg_alpha   = (-torch.log(torch.exp(-predicts["alpha_coarse"]) + torch.exp(predicts["alpha_coarse"] - 1))).mean() + OFFSET
        reg_density = (-torch.log(torch.exp(-predicts["weight_coarse"]) + torch.exp(predicts["weight_coarse"] - 1))).mean() + OFFSET
        losses["reg_alpha"] = reg_alpha
        losses["reg_density"] = reg_density
        loss += self.opt.w_reg * reg_alpha
        loss += self.opt.w_reg * reg_density

        losses["loss"] = loss
        return losses
    def debug_vis(self, predicts, targets):
        import os
        import cv2
        filedir = 'debug_loss'
        cv2.imwrite(os.path.join(filedir, 'target_rgb.jpg'), targets['rgb'])


class NeRFLoss(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

    def forward(self, predicts, targets):
        losses = {}
        loss = 0

        loss_rgb = F.mse_loss(predicts["rgb_coarse"], targets["rgb"], reduction="mean")
        loss += self.opt.w_rgb * loss_rgb
        losses["mse_loss"] = loss_rgb

        loss_alpha = F.mse_loss(predicts["alpha_coarse"], targets["alpha"])
        loss += self.opt.w_alpha * loss_alpha
        losses["loss_alpha_coarse"] = loss_alpha

        OFFSET = 0.313262
        reg_alpha   = (-torch.log(torch.exp(-predicts["alpha_coarse"]) + torch.exp(predicts["alpha_coarse"] - 1))).mean() + OFFSET
        reg_density = (-torch.log(torch.exp(-predicts["weight_coarse"]) + torch.exp(predicts["weight_coarse"] - 1))).mean() + OFFSET
        losses["reg_alpha"] = reg_alpha
        losses["reg_density"] = reg_density
        loss += self.opt.w_reg * reg_alpha
        loss += self.opt.w_reg * reg_density

        losses["loss"] = loss
        return losses

class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb_gt, rgb):
        rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }
