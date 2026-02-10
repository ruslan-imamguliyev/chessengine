from src.config.manager import ConfigurationManager
from src.logging import logger
from src.data.entity import DatasetEntity
import torch
torch.set_printoptions(threshold=10_000)
config = ConfigurationManager()

# DataTransformator(config=config).transform()
logger.info(DatasetEntity(config=config).get_data_set()[0])
