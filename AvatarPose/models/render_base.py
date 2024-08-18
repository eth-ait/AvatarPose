import torch.nn as nn
import torch
import os
from torch.utils.cpp_extension import load
import hydra
from instant_avatar.models.structures.utils import Rays

cuda_dir = os.path.join(hydra.utils.to_absolute_path('instant_avatar/renderers'), "cuda")
raymarch_kernel = load(name='raymarch_kernel',
                       extra_cuda_cflags=[],
                       sources=[f'{cuda_dir}/raymarcher.cpp',
                                f'{cuda_dir}/raymarcher.cu'])

def composite(sigma_vals, dists, thresh=0):
    # 0 (transparent) <= alpha <= 1 (opaque)
    tau = torch.relu(sigma_vals) * dists
    alpha = 1.0 - torch.exp(-tau)
    if thresh > 0:
        alpha = torch.where(alpha>=thresh, alpha, torch.zeros_like(alpha))
    # transimittance = torch.cat([torch.ones_like(alpha[..., 0:1]),
                                # torch.exp(-torch.cumsum(tau, dim=-1))], dim=-1)
    transimittance = torch.cat([torch.ones_like(alpha[..., :1]),
                                torch.cumprod(1 - alpha + 1e-10, dim=-1)], dim=-1)
    w = alpha * transimittance[..., :-1]
    return w, transimittance


def agg_pred(pred, bg_color):
    # sort all values in pred
    _, indices = torch.sort(pred['z_vals'], dim=-1) # (n_rays，n_samples)
    ind_0 = torch.arange(indices.shape[0], device=indices.device).unsqueeze(-1).expand_as(indices)
    # ind_0 = torch.zeros_like(indices, device=indices.device)
    # ind_0 = ind_0 + torch.arange(0, indices.shape[0], device=indices.device).reshape(-1, 1)
    pred_sorted = {key: val[ind_0, indices] for key, val in pred.items()}
    
    z_vals = pred_sorted['z_vals']
    if z_vals.shape[-1] == 1:
        dists = z_vals.clone()
    else:
        dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    weights, transmittance = composite(pred_sorted['sigma_vals'].reshape(z_vals.shape), dists, thresh=0)
    no_hit = transmittance[..., -1]
    depth = (weights * z_vals).sum(dim=-1)
    instance = (weights[..., None] * pred_sorted['instance_vals']).sum(dim=-2) # (n_ray, n_samples, 2)
    color = (weights[..., None] * pred_sorted['rgb_vals']).sum(dim=-2)
    if bg_color is not None:
        bg_color = bg_color.view(-1, 3)
        color = color + no_hit[..., None] * bg_color
    else:
        color = color + no_hit[..., None]
    return {
        'color': color,
        'depth': depth,
        'instance': instance,
        'weights': weights
    }

def expand_mask(ret, mask, bg_color):
    for key, val in ret.items():
        val_ = torch.zeros((mask.shape[0], *val.shape[1:]), device=val.device, dtype=val.dtype)
        if key == 'color':
            val_[~mask] = bg_color[~mask]
        val_[mask] = val
        ret[key] = val_
    
    return ret

def adjust_distance(t_0, t, d, i, n):
    l = torch.ceil((t - t_0 - i * d / n)/d)
    delta_t = l * d + i * d / n + t_0 - t
    t += delta_t
    return t

class BaseRenderer(nn.Module):
    def __init__(self, net, chunk, white_bkgd, use_occupancy, N_samples,
        render_layer=False,
        # return_raw=False, 
        return_extra=False, use_canonical=False):
        super().__init__()
        self.net = net
        self.chunk = chunk
        self.white_bkgd = white_bkgd
        self.use_occupancy = use_occupancy
        self.N_samples = N_samples
        self.return_extra = return_extra
        self.use_canonical = use_canonical
        self.render_layer = render_layer
        if use_canonical:
            self.net.use_canonical = use_canonical

    def forward(self, datas, names, use_noise=False, eval_mode=False, layered=False):
        assert(~(use_noise and eval_mode))
        if not eval_mode:
            if layered:
                results = self.forward_train_layered(datas, names, use_noise)
        else:
            if layered:
                results = self.forward_test_layered(datas, names)
        return results
    
    def adjust_rays(self, rays_dict, names, n_samples):
        n = len(names)
        d = 2/n_samples # only when the distance between near and far are fixed as 2
        rays_0 = rays_dict[names[0]]
        for i in range(1, n):
            name = names[i]
            rays = rays_dict[name]
            rays.near = adjust_distance(rays_0.near, rays.near, d, i, n)
            rays.far = adjust_distance(rays_0.far, rays.far, d, i, n)
            rays_dict[name] = rays
        return rays_dict
    
        
    def forward_train_layered(self, datas, names, use_noise):
        rays_dict = {}
        models = {name: self.net.get_model(name) for name in names}
        
        for name in names:
            rays = Rays(o=datas["rays_o"], d=datas["rays_d"], near=datas[name]["near"], far=datas[name]["far"])
            models[name].deformer.transform_rays_w2s(rays)
            rays_dict[name] = rays
            
        rays_dict = self.adjust_rays(rays_dict, names, models[name].renderer.MAX_SAMPLES)
        pred = {'rgb_vals':[], 'sigma_vals':[], 'z_vals':[], 'instance_vals':[], 'pts':[]} 
        bg_color = datas.get('bg_color', None)
          
        for name in names:
            model = models[name]
            rays = rays_dict[name]
            ret = model(rays, bg_color, eval_mode=False, use_noise=use_noise, layer=True)
            ret['instance_vals'] = torch.zeros((*ret['z_vals'].shape, len(names)), dtype=ret['z_vals'].dtype, device=ret['z_vals'].device)
            ret['instance_vals'][..., names.index(name)] = 1.
            for key in ret.keys():
                pred[key].append(ret[key])
        
        all_collision = self.cal_collision(names, pred, rays_dict)
        # caoncatenate all the pred values
        for key in pred.keys():
            pred[key] = torch.cat(pred[key], dim=1)

        ret = agg_pred(pred, bg_color)

        return {
                'rgb': ret['color'].reshape(rays.o.shape),
                'depth': ret['depth'].reshape(rays.near.shape),
                'instance': ret['instance'].reshape(*rays.near.shape, -1), 
                'alpha': (ret['weights'].sum(-1)).reshape(rays.near.shape),
                'weight': ret['weights'].reshape(*rays.near.shape, -1) ,
                'collision': all_collision  
                }
    
    def forward_test_layered(self, datas, names):
        maxlen = datas['rays_d'].reshape(-1, 3).shape[0]
        retlist = []
        pred = {}
        for bn in range(0, maxlen, self.chunk):
            start, end = bn, min(bn + self.chunk, maxlen)
            ret = self.test_layer_batch(datas, names, start, end)  
            # if bn == chunk * 14:
            #     ret['rgb'] = torch.ones_like(ret['rgb'])
            #     p1 = 500 * 940 + 500 - 14 * 16384 *2
            #     p2 = 500 * 940 + 450 - 14 * 16384 *2
            #     ret['rgb'][p1] = torch.tensor([0, 0, 0])
            #     ret['rgb'][p2] = torch.tensor([0, 0, 0])         
            if ret is not None:
                retlist.append(ret)
        for key in retlist[0].keys():
            if key == 'depth' or key == 'alpha':
                pred[key] = torch.cat([ret[key] for ret in retlist], dim=0).reshape(datas['rays_d'].shape[:-1])
            elif key == 'weight':
                pred[key] = [ret[key] for ret in retlist]
                if len(pred[key]) < 5:
                    pred[key] = torch.cat(pred[key], dim=0).reshape(*datas['rays_d'].shape[:-1], -1)
            elif key == 'collision':
                if retlist[0][key] is None:
                    pred[key] = None
                else:
                    pred[key] = torch.cat([ret[key] for ret in retlist], dim=0)
            else:
                pred[key] = torch.cat([ret[key] for ret in retlist], dim=0).reshape(*datas['rays_d'].shape[:-1], -1)
        return pred

    def cal_collision(self, names, pred, rays_dict):
        all_collision = None
        names_cnt = len(names)
        if names_cnt > 1:
            all_collision = []
            inf_tensor = torch.Tensor([1e10]).to(pred['sigma_vals'][0])
            for i in range(names_cnt-1):
                sigma_vals = pred['sigma_vals'][i]
                mask = sigma_vals>0
                density = [sigma_vals[mask]]
                z_vals = pred['z_vals'][i]
                dists = z_vals[..., 1:] - z_vals[..., :-1] if z_vals.shape[-1]>1 else z_vals.clone()
                dists = torch.cat([dists, inf_tensor.expand_as(dists[..., :1])],-1)[mask]
                for j in range(names_cnt):
                    if j == i:
                        continue
                    rays_new = rays_dict[names[j]] # o, d:(1, ray_num, 3)
                    pts_new = pred['z_vals'][i][..., None] * rays_new.d.reshape(-1,3)[:, None] + rays_new.o.reshape(-1, 3)[:, None]
                    model = self.net.get_model(names[j])
                    density.append(model.cal_density(pts_new[mask]))
                    
                density = torch.stack(density, dim=-1)
                # density = 1.0 - torch.exp(-torch.relu(density) * dists[..., None])
                density = 1.0 - torch.exp(-torch.relu(density) * 0.01)
                collision = density.prod(-1)
                all_collision.append(collision[collision > 0])
            all_collision = torch.cat(all_collision, dim=-1)
        return all_collision
    
    def test_layer_batch(self, datas, names, start, end):
        mask_at_box_union = []
        rays_dict = {}
        for name in names:
            rays = Rays(o=datas['rays_o'].reshape(-1, 3)[start:end], d=datas['rays_d'].reshape(-1, 3)[start:end], near=datas[name]['near'].reshape(-1)[start:end], far=datas[name]['far'].reshape(-1)[start:end])
            model = self.net.get_model(name)
            mask_at_box = model.deformer.transform_rays_w2s(rays).reshape(-1)
            mask_at_box_union.append(mask_at_box)
            rays_dict[name] = rays
        rays_dict = self.adjust_rays(rays_dict, names, model.renderer.MAX_SAMPLES)
        pred = {'rgb_vals':[], 'sigma_vals':[], 'z_vals':[], 'instance_vals': [], 'pts': []}
        for name in names:
            model = self.net.get_model(name)
            rays = rays_dict[name]
            mask_at_box = mask_at_box_union[names.index(name)]
            if torch.sum(mask_at_box) == 0:
                continue
            bg_color = datas.get('bg_color', None).reshape(-1, 3)[start:end]
            ret = model(rays, bg_color[mask_at_box], eval_mode=True, use_noise=False, layer=True)
            ret['instance_vals'] = torch.zeros((*ret['z_vals'].shape, len(names)), dtype=ret['z_vals'].dtype, device=ret['z_vals'].device)
            ret['instance_vals'][..., names.index(name)] = 1.
            for key in ret.keys():
                val = torch.zeros(*mask_at_box.shape, *ret[key].shape[1:], device=datas['rays_o'].device)
                val[mask_at_box] = ret[key]
                pred[key].append(val)

        all_collision = self.cal_collision(names, pred, rays_dict)
        
        mask_at_box_union = torch.stack(mask_at_box_union)
        mask_at_box_union = torch.max(mask_at_box_union, dim=0)[0]
        
        if mask_at_box_union.sum() == 0:
            pred_agg = {
                'rgb': torch.ones(*mask_at_box_union.shape, 3, dtype=torch.float32, device=datas['rays_o'].device),
                'depth': torch.zeros(*mask_at_box_union.shape, dtype=torch.float32, device=datas['rays_o'].device),
                'alpha': torch.zeros(*mask_at_box_union.shape, dtype=torch.float32, device=datas['rays_o'].device),
                'weight': torch.zeros(*mask_at_box_union.shape, 1, dtype=torch.float32, device=datas['rays_o'].device),  
                'instance': torch.zeros(*mask_at_box_union.shape, len(names), dtype=torch.float32, device=datas['rays_o'].device),
                'collision': all_collision
            }  
            return pred_agg 
        else:
            for key in pred.keys():
                pred[key] = torch.cat(pred[key], dim=1)[mask_at_box_union]
            ret = agg_pred(pred, bg_color[mask_at_box_union])
            ret = expand_mask(ret, mask_at_box_union, bg_color)
            return {
                'rgb': ret['color'],
                'depth': ret['depth'],
                'alpha': ret['weights'].sum(-1),
                'weight': ret['weights'],
                'instance': ret['instance'],
                'collision': all_collision
            }    