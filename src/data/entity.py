import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from src.config.manager import ConfigurationManager
import os
from src.constants import DEVICE
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

        train_xpath = os.path.join(self.config.input_path,
                                        "train_" + self.config.feature_filename)
        train_ypath = os.path.join(self.config.input_path,
                                        "train_" + self.config.target_filename)
        test_xpath = os.path.join(self.config.input_path,
                                        "test_" + self.config.feature_filename)
        test_ypath = os.path.join(self.config.input_path,
                                        "test_" + self.config.target_filename)

        test_num_samples = train_num_samples = self.config.num_samples

        if not self.config.num_samples:
            train_num_samples = os.path.getsize(train_xpath) // (18 * 8)
            
            logger.info("num_samples not specified for train set, using full dataset of length: {}".format(train_num_samples))

            test_num_samples = os.path.getsize(test_xpath) // (18 * 8)

            logger.info("num_samples not specified for test set, using full dataset of length: {}".format(test_num_samples))


        train_full = ChessDataset(train_xpath, train_ypath, train_num_samples)
        self.test_full = ChessDataset(test_xpath, test_ypath, test_num_samples)

        shuffled_indices = np.random.permutation(train_num_samples)

        self.train_set = Subset(train_full, shuffled_indices[:int(train_num_samples * (1 - self.config.val_split))])
        self.val_set = Subset(train_full, shuffled_indices[int(train_num_samples * (1 - self.config.val_split)):])

    
    def get_data_loader(self, mode: str) -> DataLoader:
        """
        Returns a DataLoader according to the given mode (train, val, test)
        
        :param mode: One of "train", "val", or "test" to specify which dataset to load
        :type mode: str
        :return: DataLoader for the specified dataset
        :rtype: DataLoader
        """
        logger.info("Creating DataLoader for ChessDataset.")
        
        return DataLoader(
            self.get_data_set(mode=mode),
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle if mode == "train" else False,
            num_workers=self.config.num_workers,
            pin_memory=DEVICE == "cuda"
        )
    
    def get_data_set(self, mode: str) -> ChessDataset:
        """
        Returns a ChessDataset according to the given mode (train, val, test)
        
        :param mode: One of "train", "val", or "test" to specify which dataset to load
        :type mode: str
        :return: ChessDataset for the specified dataset
        :rtype: ChessDataset
        """
        match mode:
            case "train":
                return self.train_set
            case "val":
                return self.val_set
            case "test":
                return self.test_full
            case _:
                raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'val', or 'test'.")