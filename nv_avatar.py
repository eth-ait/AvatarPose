import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import logging
import hydra
from omegaconf import OmegaConf
import time

@hydra.main(config_path="./confs", config_name="opt_avatar_smpl")
def main(opt):
    start = time.time()
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/",
        filename="epoch={epoch:04d}", # -val_psnr={val/psnr:.1f}",
        auto_insert_metric_name=False,
        save_last=True,
        **opt.checkpoint
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()
    opt.model.opt.vis_split = 'nv'
    pl_logger = TensorBoardLogger("tensorboard", name="default", version=0)
    pl_profiler = AdvancedProfiler("profiler", "advance_profiler")
    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    print('data module loaded')
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    checkpoints = sorted(glob.glob(f"checkpoints_{opt.model.opt.vis_log_name}/*.ckpt"))
    checkpoint_path = checkpoints[-1]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint found')
    trainer = pl.Trainer(gpus=1,
                         accelerator="gpu",
                         callbacks=[checkpoint_callback, lr_monitor],
                         num_sanity_val_steps=0,  # disable sanity check
                         logger=pl_logger,
                         **opt.train)


    result = trainer.test(model, ckpt_path=checkpoint_path, verbose=False)
    end = time.time()
    time_spent = end - start
    print('time spent: ', time_spent)

if __name__ == "__main__":
    main()