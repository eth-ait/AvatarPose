from ..models.structures.density_grid import DensityGrid
import torch
from torch.utils.cpp_extension import load
import os
import time
from AvatarPose.models.render_base import expand_mask

import logging
logger = logging.getLogger("nerf")
logger.propagate = False
handler = logging.FileHandler("nerf.log")
logger.addHandler(handler)

cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
raymarch_kernel = load(name='raymarch_kernel',
                       extra_cuda_cflags=[],
                       sources=[f'{cuda_dir}/raymarcher.cpp',
                                f'{cuda_dir}/raymarcher.cu'])

def stratified_sampling(N, step_size):
    device = step_size.device
    z = torch.arange(N, device=device) * step_size[..., None]
    z += torch.rand_like(z) * step_size[..., None]
    return z

def composite(sigma_vals, dists, thresh=0):
    # 0 (transparent) <= alpha <= 1 (opaque)
    tau = torch.relu(sigma_vals) * dists
    alpha = 1.0 - torch.exp(-tau)
    if thresh > 0:
        alpha[alpha < thresh] = 0
    # transimittance = torch.cat([torch.ones_like(alpha[..., 0:1]),
                                # torch.exp(-torch.cumsum(tau, dim=-1))], dim=-1)
    transimittance = torch.cat([torch.ones_like(alpha[..., 0:1]),
                                torch.cumprod(1 - alpha + 1e-10, dim=-1)], dim=-1)
    w = alpha * transimittance[..., :-1]
    return w, transimittance

def update_nohit(sigma_vals, dists, no_hit, alive, thresh=0):
    tau = torch.relu(sigma_vals) * dists
    alpha = 1.0-torch.exp(-tau)
    if thresh > 0:
        alpha[alpha < thresh] = 0
    no_hit[alive] *= torch.prod((1-alpha+1e-10), dim=-1)
    return no_hit

def ray_aabb(o, d, bbox_min, bbox_max):
    t1 = (bbox_min - o) / d
    t2 = (bbox_max - o) / d

    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)

    near = t_min.max(dim=-1).values
    far = t_max.min(dim=-1).values
    return near, far

class Raymarcher:
    def __init__(self, MAX_SAMPLES: int, MAX_BATCH_SIZE: int, smpl_init:bool=False) -> None:
        """
        Args:
            MAX_SAMPLES: number of samples per ray
            MAX_BATCH_SIZE: max samples to evaluate per batch 
        """
        self.MAX_SAMPLES = MAX_SAMPLES
        self.MAX_BATCH_SIZE = MAX_BATCH_SIZE

        self.aabb = torch.tensor([[-1.25, -1.55, -1.25],
                                  [ 1.25,  0.95,  1.25]]).float().cuda()
        self.density_grid_test = DensityGrid(64)
        self.smpl_init = smpl_init
    
    def initialize(self, N):
        if self.smpl_init:
            self.density_grid_train_all = [DensityGrid(64, self.aabb, smpl_init=True).cuda() for _ in range(N)]
        else:
            self.density_grid_train_all = [DensityGrid(64, self.aabb).cuda()]

    def __call__(self, rays, model, eval_mode=True, noise=0, bg_color=None, layer=False):
        if eval_mode:
            if layer:
                return self.render_test_layer(rays, model, bg_color)
            return self.render_test(rays, model, bg_color)
        else:
            if layer:
                return self.render_layer_train(rays, model, noise)
            return self.render_train(rays, model, noise, bg_color)
    
    @property
    def density_grid_train(self):
        # return self.density_grid_train_all[min(self.idx, len(self.density_grid_train_all) - 1)]
        return self.density_grid_train_all[len(self.density_grid_train_all) - 1]
    

    @torch.no_grad()
    def render_test(self, rays, model, bg_color):
        device = rays.o.device

        rays_o = rays.o.view(-1, 3)
        rays_d = rays.d.view(-1, 3)
        near = rays.near.view(-1)
        far = rays.far.view(-1)
        N = rays_o.shape[0]

        # property of pixels
        color = torch.zeros(N, 3, device=device)
        depth = torch.zeros(N, device=device)
        no_hit = torch.ones(N, device=device)
        counter = torch.zeros_like(depth)

        # alive indices
        alive = torch.arange(N, device=device)
        step_size = (far - near) / self.MAX_SAMPLES
        # in case of too samll step size
        step_size = torch.clamp(step_size, min=0.007812)

        density_grid = self.density_grid_test
        offset = density_grid.min_corner
        scale = density_grid.max_corner - density_grid.min_corner
        k = 0
        while k < self.MAX_SAMPLES:
            N_alive = len(alive)
            if N_alive == 0: break

            N_step = max(min(self.MAX_BATCH_SIZE // N_alive, self.MAX_SAMPLES), 1)
            pts, d_new, z_new = raymarch_kernel.raymarch_test(rays_o, rays_d, near, far, alive,
                                                              density_grid.density_field, scale, offset,
                                                              step_size, N_step)
            mask = d_new > 0
            counter[alive] += mask.sum(dim=-1)
            rgb_vals = torch.zeros_like(pts, dtype=torch.float32)
            sigma_vals = torch.zeros_like(rgb_vals[..., 0], dtype=torch.float32)
            if mask.any():
                rgb_vals[mask], sigma_vals[mask] = model(pts[mask], None)

            raymarch_kernel.composite_test(rgb_vals, sigma_vals, d_new, z_new, alive,
                                           color, depth, no_hit, 0.01)
            alive = alive[(no_hit[alive] > 1e-4) & (z_new[:, -1] > 0)]
            k += N_step
        if bg_color is not None:
            bg_color = bg_color.view(-1, 3)
            color = color + no_hit[..., None] * bg_color
        else:
            color = color + no_hit[..., None]
        # depth = depth + no_hit * 1e6
        return {
            "rgb_coarse": color.view(rays.o.shape),
            "depth_coarse": depth.view(rays.near.shape),
            "alpha_coarse": (1-no_hit).view(rays.near.shape),
            "counter_coarse": counter.view(rays.near.shape)
        }

    def render_train(self, rays, model, noise, bg_color):
        rays_o = rays.o.view(-1, 3)
        rays_d = rays.d.view(-1, 3)
        near = rays.near.view(-1)
        far = rays.far.view(-1)
        N_step = self.MAX_SAMPLES

        step_size = (far - near) / N_step
        # in case of too small step size
        step_size = torch.maximum(step_size, torch.tensor(5e-3).to(near))

        density_grid = self.density_grid_train
        offset = density_grid.min_corner
        scale = density_grid.max_corner - density_grid.min_corner

        z_vals = raymarch_kernel.raymarch_train(rays_o, rays_d, near, far,
                                                density_grid.density_field, scale, offset,
                                                step_size, N_step)
        mask = z_vals > 0
        z_vals = z_vals + torch.rand_like(z_vals) * step_size[:, None]
        pts = z_vals[..., None] * rays_d[:, None] + rays_o[:, None]

        rgb_vals = torch.zeros_like(pts, dtype=torch.float32)
        sigma_vals = -torch.ones_like(rgb_vals[..., 0], dtype=torch.float32) * 1e3
        if mask.sum() > 0:
            rgb_vals[mask], sigma_vals[mask] = model(pts[mask], None)
        if noise > 0:
            sigma_vals = sigma_vals + noise * torch.randn_like(sigma_vals)
            
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)],-1)
        dists[dists<0] = 0
        # dists = torch.ones_like(sigma_vals) * step_size[:, None]
        weights, transmittance = composite(sigma_vals.view(z_vals.shape), dists, thresh=0)
        no_hit = transmittance[..., -1]

        color = (weights[..., None] * rgb_vals.view(pts.shape)).sum(dim=-2)
        if bg_color is not None:
            bg_color = bg_color.reshape(-1, 3)
            color = color + no_hit[..., None] * bg_color
        else:
            color = color + no_hit[..., None]

        depth = (weights * z_vals).sum(dim=-1)
        return {
            "rgb_coarse": color.view(rays.o.shape),
            "depth_coarse": depth.view(rays.near.shape),
            "alpha_coarse": (weights.sum(-1)).view(rays.near.shape),
            "weight_coarse": weights.view(*rays.near.shape, -1),
        }
    
    def render_layer_train(self, rays, model, noise):
        rays_o = rays.o.view(-1, 3)
        rays_d = rays.d.view(-1, 3)
        near = rays.near.view(-1)
        far = rays.far.view(-1)
        N_step = self.MAX_SAMPLES

        step_size = (far - near) / N_step
        # in case of too small step size
        step_size = torch.maximum(step_size, torch.tensor(5e-3).to(near))

        density_grid = self.density_grid_train
        offset = density_grid.min_corner
        scale = density_grid.max_corner - density_grid.min_corner

        z_vals = raymarch_kernel.raymarch_train(rays_o, rays_d, near, far,
                                                density_grid.density_field, scale, offset,
                                                step_size, N_step)
        sum_z = z_vals.sum(dim=0)
        z_vals = z_vals[:, sum_z > 0]
        mask = z_vals > 0
        z_vals = z_vals + torch.rand_like(z_vals) * step_size[:, None]
        z_vals[~mask] = 0
        pts = z_vals[..., None] * rays_d[:, None] + rays_o[:, None]

        rgb_vals = torch.zeros_like(pts, dtype=torch.float32)
        sigma_vals = -torch.ones_like(rgb_vals[..., 0], dtype=torch.float32) * 1e3
        if mask.sum() > 0:
            rgb_vals[mask], sigma_vals[mask] = model(pts[mask], None)
        if noise > 0:
            sigma_vals += noise * torch.randn_like(sigma_vals)
        return {
            'z_vals': z_vals,
            'rgb_vals': rgb_vals,
            'sigma_vals': sigma_vals,
            'pts': pts
        }

    def render_test_layer(self, rays, model, bg_color):
        device = rays.o.device

        rays_o = rays.o.view(-1, 3)
        rays_d = rays.d.view(-1, 3)
        near = rays.near.view(-1).clone()
        far = rays.far.view(-1)
        # near, far = ray_aabb(rays_o, rays_d, self.density_grid.min_corner, self.density_grid.max_corner)
        N = rays_o.shape[0]

        # property of pixels
        no_hit = torch.ones(N, device=device)
        counter = torch.zeros(N, device=device)
        
        # depth, rgb, sigma
        pred = {
            'z_vals': [],
            'rgb_vals':[],
            'sigma_vals':[],
            'pts': [],
        }
        # alive indices
        alive = torch.arange(N, device=device)
        step_size = (far - near) / self.MAX_SAMPLES
        # in case of too samll step size
        # step_size = torch.clamp(step_size, min=0.007812)

        density_grid = self.density_grid_test
        offset = density_grid.min_corner
        scale = density_grid.max_corner - density_grid.min_corner
        k = 0
        while k < self.MAX_SAMPLES:
            N_alive = len(alive)
            if N_alive == 0: break

            N_step = max(min(self.MAX_BATCH_SIZE // N_alive, self.MAX_SAMPLES-k), 1)
            pts, d_new, z_new = raymarch_kernel.raymarch_test(rays_o, rays_d, near, far, alive,
                                                              density_grid.density_field, scale, offset,
                                                              step_size, N_step)
            counter[alive] += (d_new > 0).sum(dim=-1)
            mask = d_new > 0
            rgb_vals = torch.zeros_like(pts, dtype=torch.float32)
            sigma_vals = torch.zeros_like(rgb_vals[..., 0], dtype=torch.float32)
            if mask.any():
                rgb_vals[mask], sigma_vals[mask] = model(pts[mask], None)
                
            # concat all the intermediate variables
            ret = {
                'z_vals': z_new,
                'sigma_vals': sigma_vals,
                'rgb_vals': rgb_vals,
                'pts': pts,
            }
            for key, val in ret.items():
                val_ = torch.zeros((N, *val.shape[1:]), device=device, dtype=val.dtype)
                val_[alive] = val
                ret[key] = val_
            
            for key in pred.keys():
                pred[key].append(ret[key])
            no_hit = update_nohit(sigma_vals, d_new, no_hit, alive, thresh=0.01)
            ## when ray hit surface, stop marching
            # alive = alive[(no_hit[alive] > 1e-4) & (z_new[:, -1] > 0)]
            ## when ray hit surface, continue marching; when ray stop hit anything, stop marching
            alive = alive[(z_new[:, -1] > 0)]
            k += N_step
        for key in pred.keys():
            pred[key] = torch.cat(pred[key], dim=1)

        return pred