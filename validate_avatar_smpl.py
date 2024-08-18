import glob
import os
import torch
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import TQDMProgressBar

@hydra.main(config_path="./confs", config_name="opt_avatar_smpl")
def main(opt):
    opt.model.opt.vis_split = 'val'
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints_{opt.model.opt.vis_log_name}/",
        filename="epoch={epoch:04d}", # -val_psnr={val/psnr:.1f}",
        auto_insert_metric_name=False,
        save_last=True,
        **opt.checkpoint
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()
    
    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    checkpoints = sorted(glob.glob(f"checkpoints_{opt.model.opt.vis_log_name}/*.ckpt"))
    if len(checkpoints)>0:
        checkpoint = torch.load(checkpoints[-1])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('No checkpoints found.')
        return
        
    trainer = pl.Trainer(gpus=[0,],
                         accelerator="gpu",
                         callbacks=[checkpoint_callback, lr_monitor, TQDMProgressBar(refresh_rate=1)],
                         num_sanity_val_steps=0,  # disable sanity check
                         **opt.train)


    result = trainer.validate(model, verbose=False)   
    

if __name__ == "__main__":
    main()