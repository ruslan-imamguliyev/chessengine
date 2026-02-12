import torch.nn as nn
from src.config.manager import ConfigurationManager


class ModelManager:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_model_config()


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