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
    val_split: float


@dataclass(frozen=True)
class DatasetEntityConfig:
    input_path: str
    batch_size: int
    shuffle: bool
    num_workers: int


@dataclass(frozen=True)
class ModelConfig:
    current_model: str
    available_models: dict


@dataclass(frozen=True)
class ModelTrainerConfig:
    learning_rate: float
    weight_decay: float
    checkpoint_dir: str
    num_epochs: int
    early_stopping_patience: int
    beta: float


@dataclass(frozen=True)
class EngineConfig:
    model_strategy: str
    depth: int
    tt_size_mb: int
    time_limit: int
    book_path: str
    syzygy_path: str
    num_workers: int
    root_parallel_min_depth: int
    num_simulations: int
    leaf_parallelism: int
    puct_c: float
    prior_temperature: float


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
            val_split=config['val_split']
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
            batch_size=config['batch_size'],
            shuffle=config['shuffle'],
            num_workers=config['num_workers']
        )

    def get_model_config(self) -> ModelConfig:
        current_model = self.config['models']['current_model']
        available_models = self.config['models']['available_models']

        return ModelConfig(
            current_model=current_model,
            available_models=available_models
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config['model_trainer']

        return ModelTrainerConfig(
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            checkpoint_dir=config['checkpoint_dir'],
            num_epochs=config['num_epochs'],
            early_stopping_patience=config['early_stopping_patience'],
            beta=config['beta']
        )

    def get_engine_config(self) -> EngineConfig:
        config = self.config['engine']

        return EngineConfig(
            model_strategy=config['model_strategy'],
            depth=config['depth'],
            tt_size_mb=config['tt_size_mb'],
            time_limit=config['time_limit'],
            book_path=config['book_path'],
            syzygy_path=config['syzygy_path'],
            num_workers=config.get('num_workers', 1),
            root_parallel_min_depth=config.get('root_parallel_min_depth', 3),
            num_simulations=config.get('num_simulations', 600),
            leaf_parallelism=config.get('leaf_parallelism', 8),
            puct_c=config.get('puct_c', 1.4),
            prior_temperature=config.get('prior_temperature', 0.7)
        )
