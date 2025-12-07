from src.utils import read_yaml
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    type: str
    source: str
    install_path: str


class ConfigurationManager:
    def __init__(self,
                 config_path: str
                 ):
        self.config = read_yaml(config_path)
    
    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        return DataIngestionConfig(
            type=config.ingestion_type,
            source=config.source,
            install_path=config.install_path
        )