from src.utils import read_yaml
from src.constants import PATH_TO_CONFIG_FILE
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    method: str
    available_methods: dict


@dataclass(frozen=True)
class DataPreprocessingConfig:
    input_path: str
    output_path: str
    output_file: str
    max_mate_value: int
    min_mate_value: int
    scaling_factor: int


@dataclass(frozen=True)
class DataTransformationConfig:
    output_path: str
    input_file: str
    test_split: float
    feature_filename: str
    target_filename: str


@dataclass(frozen=True)
class DatasetEntityConfig:
    input_path: str
    feature_filename: str
    target_filename: str
    val_split: float
    batch_size: int
    shuffle: bool
    num_workers: int
    num_samples: int


@dataclass(frozen=True)
class ModelConfig:
    current_model: str
    available_models: dict


class ConfigurationManager:
    def __init__(self,
                 config_path: str = PATH_TO_CONFIG_FILE
                 ):
        self.config = read_yaml(config_path)
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        return DataTransformationConfig(
            output_path=config['output_path'],
            input_file=config['input_file'],
            test_split=config['test_split'],
            feature_filename=config['feature_filename'],
            target_filename=config['target_filename']
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        method = self.config['data_ingestion']['method']
        available_methods = self.config['data_ingestion']['available_methods']

        return DataIngestionConfig(
            method=method,
            available_methods=available_methods
        )
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config['data_preprocessing']
        
        return DataPreprocessingConfig(
            input_path=config['input_path'],
            output_path=config['output_path'],
            output_file=config['output_file'],
            max_mate_value=config['max_mate_value'],
            min_mate_value=config['min_mate_value'],
            scaling_factor=config['scaling_factor']
        )

    def get_dataset_entity_config(self) -> DatasetEntityConfig:
        config = self.config['dataset_entity']
        
        return DatasetEntityConfig(
            input_path=config['input_path'],
            feature_filename=config['feature_filename'],
            target_filename=config['target_filename'],
            batch_size=config['batch_size'],
            shuffle=config['shuffle'],
            val_split=config['val_split'],
            num_workers=config['num_workers'],
            num_samples=config['num_samples']
        )

    def get_model_config(self) -> ModelConfig:
        current_model = self.config['models']['current_model']
        available_models = self.config['models']['available_models']

        return ModelConfig(
            current_model=current_model,
            available_models=available_models
        )
