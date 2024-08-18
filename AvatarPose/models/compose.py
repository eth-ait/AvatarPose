import torch.nn as nn
from copy import deepcopy
import hydra

class ComposedModel(nn.Module):
    def __init__(self, networks, datamodule) -> None:
        super().__init__()
        models_opt = deepcopy(networks)
        names = datamodule.train_avatarset.smpl_init.keys()
        modules = {}
        for name in names:
            models_opt['pid'] = int(name.split('_')[-1])
            model = hydra.utils.instantiate(models_opt, datamodule=datamodule, _recursive_=False)
            modules[name] = model
        self.models = nn.ModuleDict(modules)
        self.keys = list(self.models.keys())
        self.is_share = False
    
    def get_model(self, name):
        model = self.models[name]
        model.current = name
        return model

    def forward(self, pts):
        raise NotImplementedError