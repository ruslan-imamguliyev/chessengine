from src.utils import read_yaml
from src.constants import PATH_TO_CONFIG_FILE
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    method: str
    available_methods: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    output_dir: str
    shard_size: int
    input_file: str


class ConfigurationManager:
    def __init__(self,
                 config_path: str = PATH_TO_CONFIG_FILE
                 ):
        self.config = read_yaml(config_path)
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        return DataTransformationConfig(
            output_dir=config['output_dir'],
            shard_size=config['shard_size'],
            input_file=config['input_file']
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        method = self.config['data_ingestion']['method']
        available_methods = self.config['data_ingestion']['available_methods']

        return DataIngestionConfig(
            method=method,
            available_methods=available_methods
        )