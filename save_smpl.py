import glob
import os
import torch
import pytorch_lightning as pl
import hydra
from tqdm import tqdm
import numpy as np
from os.path import join
from AvatarPose.utils.utils import to_tensor, to_device, to_np
import json
from AvatarPose.utils.file_utils import write_keypoints3d

@hydra.main(config_path="./confs", config_name="opt_avatar_smpl")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")
    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    plmodel = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    plmodel.cuda()
    plmodel.eval()
    checkpoints = sorted(glob.glob(f"checkpoints_{opt.model.opt.vis_log_name}/*.ckpt"))
    checkpoint_path = checkpoints[-1]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint found')
    checkpoint = torch.load(checkpoint_path)
    plmodel.load_state_dict(checkpoint['state_dict'])
    dataloader = datamodule.val_dataloader()
    
    time_cache = []
    smpl_all = {}
    kpts_path = f'kpts_{opt.model.opt.vis_log_name}'
    smpl_path = f'smpl_{opt.model.opt.vis_log_name}'
    os.makedirs(kpts_path, exist_ok=True)
    os.makedirs(smpl_path, exist_ok=True)
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader)):
            device = plmodel.device
            batch = to_device(batch, device)
            t = batch['meta']['time_idx']
            frame = batch['meta']['frame']
            if t[0] in time_cache:
                continue
            time_cache.append(t[0])
            if batch['meta']['names'] == []:
                continue
            names_all = [name[0] for name in batch['meta']['names']]
            kpts = []
            for name in names_all:
                if name not in smpl_all.keys():
                    smpl_all[name] = {}
                    for k in ['betas', 'body_pose', 'global_orient', 'transl', 'frames']:
                        smpl_all[name][k] = []
                model = plmodel.networks.get_model(name)
                smpl_params = model.SMPL_params(t)
                model.prepare_deformer(smpl_params)
                for k in ['betas', 'body_pose', 'global_orient', 'transl']:
                    smpl_all[name][k].append(smpl_params[k].detach().cpu().numpy())
                smpl_all[name]['frames'].append(frame.detach().cpu().numpy()[0])
                kpts.append({
                    'id': name.split('_')[-1],
                    'keypoints3d': to_np(model.kpts.reshape(-1, 3)),
                })
            write_keypoints3d(join(kpts_path, f'{frame[0]:06d}.json'), kpts)
    
    for name in smpl_all.keys():
        for k in smpl_all[name].keys():
            smpl_all[name][k] = np.array(smpl_all[name][k])
    
    np.savez(join(smpl_path, 'smpl_all.npz'), **smpl_all)
    

if __name__ == "__main__":
    main()