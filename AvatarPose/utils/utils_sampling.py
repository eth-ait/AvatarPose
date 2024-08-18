import numpy as np
import cv2
import math
import torch

# def get_bounds(xyz, delta=0.05):
#     '''
#     get the min, max of 3d coordinates
#     '''
#     min_xyz = np.min(xyz, axis=0)
#     max_xyz = np.max(xyz, axis=0)
#     if isinstance(delta, list):
#         delta = np.array(delta, dtype=np.float32).reshape(1, 3)
#     min_xyz -= delta
#     max_xyz += delta
#     bounds = np.stack([min_xyz, max_xyz], axis=0)
#     return bounds.astype(np.float32)

def get_bounds(xyz, delta=0.05):
    '''
    get the min, max of 3d coordinates
    '''
    min_xyz = torch.min(xyz, dim=0).values
    max_xyz = torch.max(xyz, dim=0).values
    if isinstance(delta, list):
        delta = torch.tensor(delta, dtype=torch.float32, device=xyz.device).reshape(1, 3)
    min_xyz -= delta
    max_xyz += delta
    bounds = torch.stack([min_xyz, max_xyz], dim=0)
    return bounds.to(torch.float32)

def get_near_far(ray_o, ray_d, bounds, depth_min=0.1):
    """ get near and far

    Args:
        ray_o (np): 
        ray_d ([type]): [description]
        bounds ([type]): [description]

    Returns:
        near, far, mask_at_box
        这里的near是实际物理空间中的深度
    """
    viewdir = ray_d.clone()
    viewdir[(viewdir<1e-5)&(viewdir>-1e-10)] = 1e-5
    viewdir[(viewdir>-1e-5)&(viewdir<1e-10)] = -1e-5
    inv_dir = 1.0/viewdir
    tmin = (bounds[:1] - ray_o[:1])*inv_dir
    tmax = (bounds[1:2] - ray_o[:1])*inv_dir
    # 限定时间是增加的
    t1 = torch.minimum(tmin, tmax)
    t2 = torch.maximum(tmin, tmax)
    near = torch.max(t1, dim=-1).values
    far = torch.min(t2, dim=-1).values
    near = torch.clamp(near, min=depth_min)
    mask_at_box = near < far
    return near, far, mask_at_box


def get_bound_corners(bounds):
    '''
    get the 8 corners of the bounds in 3d
    '''
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def get_bounds_2d_mask(corners_3d_camera, K, H, W):
    '''
    get 2d mask according to 3d corners, if corners are all in front of the camera
    '''
    homo_corners = np.dot(corners_3d_camera, K.T)
    corners_2d = homo_corners[:, :2]/homo_corners[:, 2:]
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype = np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

# def get_near_far(ray_o, ray_d, bounds, depth_min = 0.1):
#     norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
#     ray_dn = ray_d/norm_d
#     ray_dn[(ray_dn<1e-5)&(ray_dn>-1e-10)] = 1e-5
#     ray_dn[(ray_dn>-1e-5)&(ray_dn<1e-10)] = -1e-5
#     inv_dir = 1.0/ray_dn
#     tmin = (bounds[:1] - ray_o[:1])*inv_dir
#     tmax = (bounds[1:2] - ray_o[:1])*inv_dir
#     t1 = np.minimum(tmin, tmax)
#     t2 = np.maximum(tmin, tmax)
#     near = np.max(t1, axis=-1)
#     far = np.min(t2, axis=-1)
#     near = np.maximum(near, depth_min)
#     mask_at_box = near < far #(H, W)
#     return near, far, mask_at_box



def get_rays(H, W, cam_info):
    # useful params
    R = cam_info['R']
    T = cam_info['T']
    invK = cam_info['invK']

    # camera origin in world coordinate
    rays_o = (-(R.T) @ T).ravel()
    # pixel in world coordinate
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), 
                       np.arange(H, dtype=np.float32),  
                       indexing='xy') # i.shape = (H, W), j.shape = (H, W)
    homo_pixel = np.stack([i, j, np.ones(i.shape)], axis = 2) # homo_pixel.shape = (H, W, 3)
    pixel_camera = np.dot(homo_pixel, invK.T)
    pixel_world = np.dot(pixel_camera-T.ravel(), R)

    # ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True) # normalized direction
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    rays_o = rays_o.astype(np.float32) # (H, W, 3)
    rays_d = rays_d.astype(np.float32) # (H, W, 3)
    return rays_o, rays_d#, pixel_camera, pixel

    
def sample_coords(bound, rate, bkgd_mask):
    '''
    sample bound in bkgd_mask with rate
    '''
    coords_all = np.argwhere(bound * bkgd_mask > 0)
    if rate == 1:
        coords = coords_all
    elif rate >= 1:
        # repeat integer part
        coords_1 = np.vstack([coords_all for _ in range(math.floor(rate))])
        if not isinstance(rate, int):
            # repeat the float part
            nsample2 = int(len(coords_all) * (rate-math.floor(rate)))
            coords_2 = coords_all[np.random.randint(0, len(coords_all), nsample2)]
            coords = np.vstack([coords_1, coords_2])
        else:
            coords = coords_1
    else:
        coords = coords_all[np.random.randint(0, len(coords_all), int(len(coords_all)*rate))]

    return coords


class HumanSample:
    def __init__(self, name, feat, rate, dilate, no_body_mask):
        self.name = name
        self.feat = feat
        self.rate_bounds = rate['rate_bounds']
        self.rate_body = rate['rate_body']
        self.vert_bounds = feat['vert_bounds']
        self.mask_body = feat['mask_body']
        self.dilate = dilate
        self.no_body_mask = no_body_mask

    def mask_rate(self, cam_info, H, W):
        '''
        get the mask and sampling rate:
        in return:
        1. if no body mask used
        bound: the mask from 3d vertice bounds
        2. if use body mask
        mask_bound is the padded bound mask of body mask
        body: mask_body
        out_body: mask in mask_bound by not in mask_body
        '''
        ret = {'bound':{}, 'body':{}, 'out_body':{}}
        if self.mask_body is None and not self.no_body_mask:
            raise ValueError('the human body does no t have a mask')
        if self.no_body_mask:
            mask_bounds = self.vert_bounds_to_mask(cam_info, H, W)
            ret['bound'] = {'mask': mask_bounds, 'rate': self.rate_bounds}  
            return ret
        mask_bounds = np.zeros((H, W), dtype = np.uint8)
        ys, xs = np.where(self.mask_body)
        padding = max(mask_bounds.shape[0]//50, 32)
        bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
        # bbox = np.array([np.maximum(bbox_temp[0], 0), 
        #                  np.maximum(bbox_temp[1], 0), 
        #                  np.minimum(bbox_temp[2], W-1), 
        #                  np.minimum(bbox_temp[3], H-1)])
        mask_bounds[bbox[1]: bbox[3], bbox[0]:bbox[2]] = True # mask bounds from 2d body mask
        mask_out_body = mask_bounds ^ self.mask_body # in the bound but outside body

        mask_body = self.mask_body.copy().astype(np.uint8)
        if self.dilate: # points near surface would not be sampled
            border = 10
            kernel = np.ones((border, border), np.uint8)
            mask_erode = cv2.erode(mask_body.copy(), kernel)
            mask_dilate = cv2.dilate(mask_body.copy(), kernel)
            mask_body[(mask_dilate-mask_erode) == 1] = 0
            mask_out_body[(mask_dilate-mask_erode) == 1] = 0
        size_body = mask_body.sum()
        size_out_body = mask_out_body.sum()
        rate_body = self.rate_body
        rate_out_body = 1-self.rate_body

        if size_body < 10 or size_out_body < 10:
            print('size_body < 10 or size_out_body < 10')
            ret['out_body'] = {'mask': mask_out_body, 'rate': rate_out_body*self.rate_bounds}

        rate_body = rate_body * (size_body+size_out_body)/size_body
        rate_out_body = rate_out_body * (size_body+size_out_body)/size_out_body
        ret['body'] = {'mask': mask_body, 'rate': rate_body*self.rate_bounds}
        ret['out_body'] = {'mask': mask_out_body, 'rate': rate_out_body*self.rate_bounds}
        return ret
    
    def sample_coords(self, mask_rate_dict, bkgd_mask):
        # when training
        coords = []
        for key, val in mask_rate_dict.items():
            if val != {}:
                coords.append(sample_coords(val['mask'], val['rate'], bkgd_mask))
        coords = np.vstack(coords)
        return coords
    
    def sample_near_far(self, ray_o, ray_d):
        '''
        get the mask for ray_o and ray_d passing the vertice bounds.
        '''
        near, far, mask_at_box = get_near_far(ray_o, ray_d, self.vert_bounds)
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)

        near = near[mask_at_box] / norm_d[mask_at_box, 0]
        far = far[mask_at_box] / norm_d[mask_at_box, 0]
        return near, far, mask_at_box

    def vert_bounds_to_mask(self, cam_info, H, W):
        '''
        get the 2d mask from 3d vertice bound:
        consider 2 cases:
        1. some part of the 3d vertice bound behind the camera
        2. all vertice bound in front of the camera
        '''
        R = cam_info['R']
        T = cam_info['T']
        K = cam_info['K']

        corners_3d_world = get_bound_corners(self.vert_bounds)
        corners_3d_camera = np.dot(corners_3d_world, R.T) + T.T
        if (corners_3d_camera[..., -1] < 0).any(): # some point behind the camera, at the edge of the image
            ray_o, ray_d, _, _ = get_rays(H, W, cam_info) # ray_d.shape = ray_o.shape = (H, W, 3)
            _, _, mask = get_near_far(ray_o, ray_d, self.vert_bounds) # get the mask of the rays
        else:
            mask = get_bounds_2d_mask(corners_3d_camera, K, H, W)
        return mask
    
def create_cameras_mean_2(cameras, camera_args):
    Told = np.stack([d['T'] for d in cameras])
    Rold = np.stack([d['R'] for d in cameras])
    Kold = np.stack([d['K'] for d in cameras])
    Cold = - np.einsum('bmn,bnp->bmp', Rold.transpose(0, 2, 1), Told)
    center = Cold.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(Cold - center, axis=1).mean()
    ymean = Rold[:, 1, 1].mean()
    xznorm = np.sqrt(1. - ymean**2)
    thetas = np.linspace(0., 2*np.pi, camera_args['allstep'])
    # 计算第一个相机对应的theta
    dir0 = Cold[0] - center[0]
    # dir0[2, 0] = 0.
    dir0[1, 0] = 0.
    dir0 = dir0 / np.linalg.norm(dir0)
    # theta change
    theta0 = np.arctan2(dir0[0,0], dir0[2,0]) + np.pi/2
    thetas += theta0
    sint = np.sin(thetas)
    cost = np.cos(thetas)
    R1 = np.stack([-sint, np.zeros_like(sint), cost]).T
    R2 = xznorm * np.stack([cost, np.zeros_like(sint), sint]).T
    R2[:, 1] = ymean
    R3 = - np.cross(R1, R2)
    Rnew = np.stack([R1, R2, R3], axis=1)
    # set locations
    
    loc = np.stack([radius * cost, np.zeros_like(sint), radius * sint], axis=1)[..., None] + center
    # loc = np.stack([radius * sint, np.zeros_like(sint), -radius * cost], axis=1)[..., None] + center
    print('[sample] camera centers: ', center[0].T[0])
    print('[sample] camera radius: ', radius)
    print('[sample] camera start theta: ', theta0)
    Tnew = -np.einsum('bmn,bnp->bmp', Rnew, loc)
    K = Kold.mean(axis=0, keepdims=True).repeat(Tnew.shape[0], 0)
    return K, Rnew, Tnew

 
def create_cameras_mean(cameras, camera_args):
    Told = np.stack([d['T'] for d in cameras])
    Rold = np.stack([d['R'] for d in cameras])
    Kold = np.stack([d['K'] for d in cameras])
    Cold = - np.einsum('bmn,bnp->bmp', Rold.transpose(0, 2, 1), Told)
    center = Cold.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(Cold - center, axis=1).mean()
    zmean = Rold[:, 2, 2].mean()
    xynorm = np.sqrt(1. - zmean**2)
    thetas = np.linspace(0., 2*np.pi, camera_args['allstep'])
    # 计算第一个相机对应的theta
    dir0 = Cold[0] - center[0]
    dir0[2, 0] = 0.
    dir0 = dir0 / np.linalg.norm(dir0)
    theta0 = np.arctan2(dir0[1,0], dir0[0,0]) + np.pi/2
    thetas += theta0
    sint = np.sin(thetas)
    cost = np.cos(thetas)
    R1 = np.stack([cost, sint, np.zeros_like(sint)]).T
    R3 = xynorm * np.stack([-sint, cost, np.zeros_like(sint)]).T
    R3[:, 2] = zmean
    R2 = - np.cross(R1, R3)
    Rnew = np.stack([R1, R2, R3], axis=1)
    # set locations
    loc = np.stack([radius * sint, -radius * cost, np.zeros_like(sint)], axis=1)[..., None] + center
    print('[sample] camera centers: ', center[0].T[0])
    print('[sample] camera radius: ', radius)
    print('[sample] camera start theta: ', theta0)
    Tnew = -np.einsum('bmn,bnp->bmp', Rnew, loc)
    K = Kold.mean(axis=0, keepdims=True).repeat(Tnew.shape[0], 0)
    return K, Rnew, Tnew

def create_center_radius(center, radius=5., up='z', ranges=[0, 360, 360], angle_x=0, **kwargs):
    center = np.array(center).reshape(1, 3)
    thetas = np.deg2rad(np.linspace(*ranges))
    st = np.sin(thetas)
    ct = np.cos(thetas)
    zero = np.zeros_like(st)
    Rotx = cv2.Rodrigues(np.deg2rad(angle_x) * np.array([1., 0., 0.]))[0]
    if up == 'z':
        center = np.stack([radius*ct, radius*st, zero], axis=1) + center
        R = np.stack([-st, ct, zero, zero, zero, zero-1, -ct, -st, zero], axis=-1)
    elif up == 'y':
        center = np.stack([radius*ct, zero, radius*st, ], axis=1) + center
        R = np.stack([
            +st,  zero,  -ct,
            zero, zero-1, zero, 
            -ct,  zero, -st], axis=-1)
    R = R.reshape(-1, 3, 3)
    R = np.einsum('ab,fbc->fac', Rotx, R)
    center = center.reshape(-1, 3, 1)
    T = - R @ center
    RT = np.dstack([R, T])
    return RT