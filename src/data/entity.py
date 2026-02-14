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
    def __init__(self, config):
        self.config = config.get_dataset_entity_config()

        train_xpath = os.path.join(self.config.input_path, f"train_{self.config.feature_filename}")
        train_ypath = os.path.join(self.config.input_path, f"train_{self.config.target_filename}")
        test_xpath = os.path.join(self.config.input_path, f"test_{self.config.feature_filename}")
        test_ypath = os.path.join(self.config.input_path, f"test_{self.config.target_filename}")
        
        total_train_samples = os.path.getsize(train_xpath) // (18 * 8) if not self.config.num_samples else self.config.num_samples
        test_num_samples = os.path.getsize(test_xpath) // (18 * 8) if not self.config.num_samples else self.config.num_samples

        val_size = int(total_train_samples * self.config.val_split)
        train_size = total_train_samples - val_size

        logger.info(f"Total train file: {total_train_samples} samples. Splitting into:")
        logger.info(f"  -> Train: {train_size}")
        logger.info(f"  -> Val:   {val_size}")

        self.train_set = ChessDataset(train_xpath, train_ypath, num_samples=train_size, start_idx=0)
        self.val_set = ChessDataset(train_xpath, train_ypath, num_samples=val_size, start_idx=train_size)
        self.test_set = ChessDataset(test_xpath, test_ypath, num_samples=test_num_samples, start_idx=0)

    def get_data_loader(self, mode: str) -> DataLoader:
        dataset = self.get_data_set(mode)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=(DEVICE == "cuda"),
            prefetch_factor=1,
            persistent_workers=True
        )
    
    def get_data_set(self, mode: str):
        match mode:
            case "train": return self.train_set
            case "val": return self.val_set
            case "test": return self.test_set
            case _: raise ValueError(f"Invalid mode: {mode}")