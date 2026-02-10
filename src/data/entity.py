import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.config.manager import ConfigurationManager
import os
from src.logging import logger


class ChessDataset(Dataset):
    def __init__(self, x_path, y_path, num_samples):
        self.X = np.memmap(x_path, dtype='uint8', mode='r', shape=(num_samples, 18, 8))
        self.Y = np.memmap(y_path, dtype='float32', mode='r', shape=(num_samples,))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        packed_features = self.X[idx]
        
        features = np.unpackbits(packed_features, axis=-1).reshape(18, 8, 8).astype(np.float32)
        
        target = torch.tensor(self.Y[idx], dtype=torch.float32)
        
        return torch.from_numpy(features), target


class DatasetEntity:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_dataset_entity_config()
    
    def get_data_loader(self) -> DataLoader:
        logger.info("Creating DataLoader for ChessDataset.")
        
        return DataLoader(
            self.get_data_set(),
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
    
    def get_data_set(self) -> ChessDataset:
        x_path = os.path.join(self.config.input_path, self.config.feature_filename)
        y_path = os.path.join(self.config.input_path, self.config.target_filename)
        
        if not self.config.num_samples:
            self.config.num_samples = os.path.getsize(x_path) // (18 * 8)
            logger.info("num_samples not specified, using full dataset of length: {}".format(self.config.num_samples))
        
        return ChessDataset(
                x_path=x_path,
                y_path=y_path,
                num_samples=self.config.num_samples
            )