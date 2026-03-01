import torch.nn as nn
import torch
from collections import OrderedDict
from src.constants import DEVICE
from src.config.manager import ConfigurationManager
import os


class ModelManager:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_model_config()
        self.checkpoints_dir = config.get_model_trainer_config().checkpoint_dir


    def get_model(self) -> nn.Module:
        current_model = self.config.current_model
        available_models = self.config.available_models

        if not current_model in available_models:
            raise ValueError(f"Model '{current_model}' is not in available models: {list(available_models.keys())}")
        
        model_type, params = available_models[current_model]['model_type'], available_models[current_model]['params']

        match model_type:
            case "resnet":
                from src.models.resnet import ChessResNet
                return ChessResNet(**params)
            case _:
                raise ValueError(f"Model '{current_model}' is not implemented yet.")
    
    def get_model_name(self) -> str:
        return self.config.current_model
    
    def load_model(self, strategy: str="latest") -> nn.Module:
        if not os.path.exists(
            os.path.join(
                os.getcwd(),
                self.checkpoints_dir
            )
        ):
            raise FileNotFoundError(f"Checkpoints directory '{self.checkpoints_dir}' does not exist.")

        checkpoints_dir = os.path.join(
                os.getcwd(),
                self.checkpoints_dir
            )
        
        if not os.path.exists(
            os.path.join(
                checkpoints_dir,
                strategy + "_checkpoint.pth"
            )
        ):
            raise FileNotFoundError(f"Checkpoint file '{strategy}_checkpoint.pth' does not exist in '{checkpoints_dir}'.")

        state_dict = torch.load(os.path.join(checkpoints_dir, strategy + "_checkpoint.pth"), map_location=DEVICE)['model_state_dict']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        
        model = self.get_model()
        model.load_state_dict(new_state_dict)
        
        del new_state_dict, state_dict
        return model