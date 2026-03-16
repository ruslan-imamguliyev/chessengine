from src.config.manager import ConfigurationManager
from src.models.compiler import ModelCompiler


if __name__ == "__main__":
    from src.models import ModelManager
    config = ConfigurationManager()
    model_compiler = ModelCompiler(config)
    model = ModelManager(config).load_model()
    model_compiler.compile(model)