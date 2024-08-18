from tqdm import tqdm
import cv2
import os
import numpy as np
from glob import glob
from AvatarPose.vis.vis_pose import vis_kpts
from AvatarPose.utils.utils import to_np, to_device, to_tensor
import argparse

def save_output_videos(image_root, pids, fps=30):
    video_root = f'{image_root}/videos'
    os.makedirs(video_root, exist_ok=True)
    roots_1 = glob(f'{image_root}/*')
    for root_1 in roots_1:
        n1 = os.path.basename(root_1)
        if n1 == 'videos':
            continue
        roots_2 = glob(f'{root_1}/*')
        for root_2 in tqdm(roots_2):
            result = os.path.basename(root_2)
            if result in ['kpts', 'rgb_err', 'rgb_cano', 'rgb']:
                if result != 'rgb_cano':
                    images = glob(f'{root_2}/*')
                    n3s = np.array([int(os.path.basename(image).split('.')[0].split('_')[-1]) for image in images])
                    indexs = np.argsort(n3s)
                    images = [images[idx] for idx in indexs]
                    frame_width, frame_height = cv2.imread(images[0]).shape[1], cv2.imread(images[0]).shape[0]
                    videoname = f'{video_root}/{result}_{n1}.mp4'
                    video_writer = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                    for image in images:
                        video_writer.write(cv2.imread(image))
                else:
                    for pid in pids:
                        images = glob(f'{root_2}/*_{pid}.png')
                        n3s = np.array([int(os.path.basename(image).split('.')[0].split('_')[-2]) for image in images])
                        indexs = np.argsort(n3s)
                        images = [images[idx] for idx in indexs]
                        frame_width, frame_height = cv2.imread(images[0]).shape[1], cv2.imread(images[0]).shape[0]
                        videoname = f'{video_root}/{result}_{n1}_{pid}.mp4'
                        video_writer = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                        for image in images:
                            video_writer.write(cv2.imread(image))
        print('saved', n1)
        
        
class Visimg:
    def __init__(self, root, task):
        self.root = root
        self.task = task # different tasks have different order of directories
        assert(self.task in ['train', 'val', 'nv'])
        self.key_lists = ['rgb', 'rgb_cano', 'instance', 'alpha', 'depth', 'kpts', 'kpts_gt']
        self.color_rgb =[
            [28/255, 255/255, 120/255], # green
            [ 28/255, 163/255, 255/255],# orange hi4d pid1
            [255/255, 120/255,  28/255],# blue hi4d pid0
            [255/255, 28/255, 180/255], # red
            [120/255, 255/255, 28/255], # yellow
            ]
        
    def set_names(self, split, epoch, cam, frame, step):
        self.split = split
        self.step = step
        if self.task == 'train':
            self.image_root = f'{self.root}/{self.split}_{frame}'
            self.n1 = cam
            self.n2 = epoch
        elif self.task == 'val':
            self.image_root = f'{self.root}/{self.split}_{epoch}'
            self.n1 = cam
            self.n2 = frame
        elif self.task == 'nv':
            self.image_root = f'{self.root}/{self.split}_{epoch}'
            self.n1 = frame
            self.n2 = cam
        
    def set_results(self, keys):
        self.keys = keys
        for key in self.keys:
            if key not in self.key_lists:
                raise ValueError('invalid result type')
        
    def saveimg(self, predicts, gts):
        predicts = to_np(predicts)
        gts = to_np(gts)
        for key in self.keys:
            if key == 'rgb':
                # visualize heatmap (blue ~ 0, red ~ 1)
                rgb = predicts[key]
                os.makedirs(f"{self.image_root}/{self.n1}/rgb", exist_ok=True)
                cv2.imwrite(f"{self.image_root}/{self.n1}/rgb/rgb_{self.n2}.png", rgb[0][..., [2, 1, 0]] * 255)
                if gts.get('rgb') is not None:
                    rgb_gt = gts[key]       
                    errmap = np.sqrt(np.square(rgb-rgb_gt).sum(-1))[0] / np.sqrt(3)
                    errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    errmap_rgb = errmap[None] / 255
                    rgbs = np.concatenate([rgb_gt, rgb, errmap_rgb], axis=2)
                    os.makedirs(f"{self.image_root}/{self.n1}/rgb_err", exist_ok=True)
                    cv2.imwrite(f"{self.image_root}/{self.n1}/rgb_err/rgb_err_{self.n2}.png", rgbs[0][..., [2, 1, 0]] * 255)
            
            if key == 'rgb_cano':
                # visualize canonical rgb
                os.makedirs(f"{self.image_root}/{self.n1}/rgb_cano", exist_ok=True)
                for name in predicts['names_all']:
                    pid = int(name.split('_')[-1])
                    # cano = np.concatenate([predicts['rgb'], predicts[name][key]], axis=2) 
                    cano = predicts[name][key]
                    cv2.imwrite(f"{self.image_root}/{self.n1}/rgb_cano/rgb_cano_{self.n2}_{pid}.png", cano[0][..., [2, 1, 0]] * 255)

            if key == 'alpha':
                # errmap alpha visualize heatmap (blue ~ 0, red ~ 1)
                alpha = predicts[key]
                alpha_rgb = np.repeat(alpha[...,None], 3, axis=-1)
                os.makedirs(f"{self.image_root}/{self.n1}/alpha", exist_ok=True)
                cv2.imwrite(f"{self.image_root}/{self.n1}/alpha/alpha_{self.n2}.png", alpha_rgb[0][..., [2, 1, 0]] * 255)
                if gts.get('alpha') is not None:
                    alpha_gt = gts[key]
                    errmap = np.abs(alpha - alpha_gt)[0]
                    errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    errmap_alpha = errmap[None] / 255
                    alphas = np.concatenate([np.repeat(alpha[...,None], 3, axis=-1), errmap_alpha], axis=2)
                    os.makedirs(f"{self.image_root}/{self.n1}/alpha_err", exist_ok=True)
                    cv2.imwrite(f"{self.image_root}/{self.n1}/alpha_err/alpha_err_{self.n2}.png", alphas[0][..., [2, 1, 0]] * 255)
            
            if key == 'instance':
                # get instance output
                instance = predicts[key]
                pids_all = np.arange(instance.shape[-1])
                colors = []
                for pid in range(len(pids_all)):
                    self.color_rgb[pid] = self.color_rgb[pid][::-1]
                    colors.append(self.color_rgb[pid])
                colors = np.array(colors)
                instance_color = instance @ colors
                os.makedirs(f"{self.image_root}/{self.n1}/instance", exist_ok=True)
                cv2.imwrite(f"{self.image_root}/{self.n1}/instance/instance_{self.n2}.png", instance_color[0][..., [2, 1, 0]] * 255)
                instance_color_argmax = colors[instance.argmax(axis=-1)]
                instance_color_argmax[instance.max(axis=-1) == 0] = np.zeros(3)
                os.makedirs(f"{self.image_root}/{self.n1}/instance_argmax", exist_ok=True)
                cv2.imwrite(f"{self.image_root}/{self.n1}/instance_argmax/instance_argmax_{self.n2}.png", instance_color_argmax[0][..., [2, 1, 0]] * 255)     
                if gts.get('pids_mask') is not None:
                    instance_gt = gts['pids_mask']
                    instance_gt_color = np.zeros_like(instance_color)
                    for pid in pids_all:
                        instance_gt_color[instance_gt == pid+1] = colors[pid]
                    instances = np.concatenate([instance_gt_color, instance_color, instance_color_argmax], axis=2)
                    os.makedirs(f"{self.image_root}/{self.n1}/instance_err", exist_ok=True)
                    cv2.imwrite(f"{self.image_root}/{self.n1}/instance_err/instance_err_{self.n2}.png", instances[0][..., [2, 1, 0]] * 255)        
            
            if key == 'depth':
                # get depth output
                depth = predicts[key]
                min_depth = 0.1
                max_depth = 5
                depth = (depth - min_depth) / (max_depth - min_depth)
                depth_rgb = (np.clip(depth, 0, 1.) * 255).astype(np.uint8)[0]
                depth_rgb = cv2.applyColorMap(depth_rgb, cv2.COLORMAP_JET)
                os.makedirs(f"{self.image_root}/{self.n1}/depth", exist_ok=True)
                cv2.imwrite(f"{self.image_root}/{self.n1}/depth/depth_{self.n2}.png", depth_rgb)
                
            if key == 'kpts':
                img_gt = vis_kpts(predicts, gts['cam_info']['P'][0], (gts['rgb'][0][..., [2, 1, 0]]*255).copy())
                # img_pred = vis_kpts(predicts, gts['cam_info']['P'][0], (predicts['rgb'][0][..., [2, 1, 0]]*255).copy())
                # kpts_img = np.concatenate([img_gt, img_pred], axis=1)
                # os.makedirs(f"{self.image_root}/{self.n1}/kpts", exist_ok=True)
                os.makedirs(f"{self.image_root}/{self.n1}/kpts_fig", exist_ok=True)
                # cv2.imwrite(f'{self.image_root}/{self.n1}/kpts/kpts_{self.n2}.png', kpts_img)
                cv2.imwrite(f'{self.image_root}/{self.n1}/kpts_fig/kpts_{self.n2}.png', img_gt)

            if key == 'kpts_gt':
                img_gt = vis_kpts(predicts, gts['cam_info']['P'][0], (gts['rgb'][0][..., [2, 1, 0]]*255).copy(), key='kpts_gt')
                img_pred = vis_kpts(predicts, gts['cam_info']['P'][0], (predicts['rgb'][0][..., [2, 1, 0]]*255).copy(), key='kpts_gt')
                kpts_img = np.concatenate([img_gt, img_pred], axis=1)
                os.makedirs(f"{self.image_root}/{self.n1}/kpts_gt", exist_ok=True)
                cv2.imwrite(f'{self.image_root}/{self.n1}/kpts_gt/kpts_gt_{self.n2}.png', kpts_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--pids', type=str, default=[0, 1])
    parser.add_argument('--video_root', type=str, default='videos')
    parser.add_argument('--video_name', type=str, default='video')
    parser.add_argument('--task', type=str, choices=['1', '2', 'output'], default='output')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    if args.task == 'output':
        save_output_videos(args.root, args.pids, args.fps)
    elif args.task == '1':
        save_video(args.root, args.video_root, args.video_name)
    elif args.task == '2':
        save_videos(args.root, args.video_root)