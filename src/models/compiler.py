from src.config.manager import ConfigurationManager
from torch.nn import Module
import torch
from src.logging import logger
import os


class ModelCompiler:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_model_compiler_config()
    
    def compile(self, model: Module) -> None:
        model.eval()
        logger.info("Compiling the model using TorchScript...")
        compiled = torch.jit.script(model)
        output_path = os.path.join(
            os.getcwd(),
            self.config.output_path
        )
        compiled.save(output_path)
        logger.info(f"Model compiled and saved to {output_path}")


if __name__ == "__main__":
    from src.models import ModelManager
    config = ConfigurationManager()
    model_compiler = ModelCompiler(config)
    model = ModelManager(config).load_model()
    model_compiler.compile(model)