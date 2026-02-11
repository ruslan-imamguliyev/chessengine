from src.config.manager import ConfigurationManager
from src.logging import logger
from src.data.transformation import DataTransformator
from src.utils import create_paths
import torch

torch.set_printoptions(threshold=10_000)

config = ConfigurationManager()
DataTransformator(config=config).transform()