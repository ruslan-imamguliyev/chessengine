import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from src.config.manager import ConfigurationManager
import os
from src.constants import DEVICE
from src.logging import logger


class ChessDataset(Dataset):
    def __init__(self, encodings, evaluations):
        self.encodings = encodings
        self.evaluations = evaluations
    
    def __len__(self):
        return len(self.evaluations)
    
    def __getitem__(self, idx):
        return self.encodings[idx], self.evaluations[idx]


class DatasetEntity:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_dataset_entity_config()
        
        train_path = os.path.join(self.config.input_path, "train.pth")
        val_path = os.path.join(self.config.input_path, "val.pth")

        train_metadata = torch.load(train_path)
        val_metadata = torch.load(val_path)

        train_num_samples = train_metadata['num_samples']
        val_num_samples = val_metadata['num_samples']

        logger.info(f"Creating train dataset with {train_num_samples} samples.")
        logger.info(f"Creating val dataset with {val_num_samples} samples.")

        self.train_set = ChessDataset(train_metadata['features_tensor'], train_metadata['targets_tensor'])
        self.val_set = ChessDataset(val_metadata['features_tensor'], val_metadata['targets_tensor'])

        del train_metadata, val_metadata

    def get_data_loader(self, mode: str) -> DataLoader:
        dataset = self.get_data_set(mode)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle and mode == "train",
            num_workers=self.config.num_workers,
            pin_memory=(DEVICE == "cuda"),
            prefetch_factor=max(self.config.num_workers // 2, 1),
            persistent_workers=True
        )
    
    def get_data_set(self, mode: str) -> ChessDataset:
        match mode:
            case "train": return self.train_set
            case "val": return self.val_set
            case _: raise ValueError(f"Invalid mode: {mode}")