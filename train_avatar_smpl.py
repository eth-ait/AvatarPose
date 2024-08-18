import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import hydra
from omegaconf import OmegaConf
from AvatarPose.utils.utils import check_max_epoch, find_dict_diff
import time


@hydra.main(config_path="./confs", config_name="opt_avatar_smpl")
def main(opt):

    start = time.time()
    
    opt.model.opt.vis_split = 'train'
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    print('data module loaded')
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    print('model loaded')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints_{opt.model.opt.vis_log_name}/",
        filename="epoch={epoch:04d}", # -val_psnr={val/psnr:.1f}",
        auto_insert_metric_name=False,
        save_last=True,
        **opt.checkpoint)
    lr_monitor = pl.callbacks.LearningRateMonitor()
    pl_logger = TensorBoardLogger(f"tensorboard_{opt.model.opt.vis_log_name}", name="default")
    trainer = pl.Trainer(gpus=[0,],
                         accelerator="gpu",
                         callbacks=[checkpoint_callback, lr_monitor],
                         num_sanity_val_steps=0,  # disable sanity check
                         logger=pl_logger,
                         reload_dataloaders_every_epoch=True,
                         **opt.train)

    check_max_epoch(opt)

    checkpoints_avatar_smpl = sorted(glob.glob(f"checkpoints_{opt.model.opt.vis_log_name}/*.ckpt"))
    checkpoints_base = sorted(glob.glob(f"checkpoints_{opt.model.opt.base}/*.ckpt"))

    if len(checkpoints_avatar_smpl) > 0 and opt.resume:
        opt_exist = OmegaConf.load(f'config_{opt.model.opt.vis_log_name}.yaml')
        if opt == opt_exist:
            print("Resume from", checkpoints_avatar_smpl[-1])
            trainer.fit(model, ckpt_path=checkpoints_avatar_smpl[-1])
        else:
            print("warning: resume configuration different")
            find_dict_diff(opt, opt_exist)
            trainer.fit(model, ckpt_path=checkpoints_avatar_smpl[-1])
        
    elif len(checkpoints_base) > 0:
        checkpoint_base = torch.load(checkpoints_base[-1])
        if opt.model.opt.load_avatar:
            print("Resume avatar state from", checkpoints_base[-1])
            model.load_avatar_weights(checkpoint_base['state_dict'])
        if opt.model.opt.load_smpl:
            print("Resume smpl state from", checkpoints_base[-1])
            model.load_smpl_weights(checkpoint_base['state_dict'])
        if not opt.model.opt.load_avatar and not opt.model.opt.load_smpl:
            print("The base checkpoint is not used", checkpoints_base[-1], 'please set opt.model.opt.load_avatar or opt.model.opt.load_smpl to True')
            return 0
        trainer.fit(model)
        
    else:  
        print("New config, saving configs.")
        OmegaConf.save(opt, f"config_{opt.model.opt.vis_log_name}.yaml")
        trainer.fit(model)
    
    end = time.time()
    time_spent = end - start
    print('time spent: ', time_spent)

    
if __name__ == "__main__":
    main()
