import numpy as np
import tqdm
from src.utils import fen_to_tensor
from src.logging import logger
from math import ceil
import os
import torch
from src.utils import create_paths
from src.config.manager import ConfigurationManager
import pandas as pd


class DataTransformator:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_data_transformation_config()


    def transform(self) -> None:
        
        create_paths(self.config.output_path)
        
        logger.info("Reading the full dataset.")

        # TODO: delete nrows
        df = pd.read_csv(self.config.input_file, nrows=1_000_000)

        logger.info("Shuffling and splitting the dataset into train and test sets.")

        shuffled_indices = np.random.permutation(len(df))
        test_split = int(len(df) * self.config.test_split)
        train_df = df.iloc[shuffled_indices[test_split:]]
        test_df = df.iloc[shuffled_indices[:test_split]]

        self.save(train_df, "train")
        self.save(test_df, "test")

        logger.info("Successfully transformed the dataset and saved to .pt files.")

    def save(
            self,
            df: pd.DataFrame,
            prefix: str
        ) -> None:

        num_samples = len(df)
        logger.info(f"Transforming {num_samples} entires of the {prefix} set.")

        output_filepath = os.path.join(
            self.config.output_path,
            prefix + ".pt"
        )
        
        features = np.zeros((num_samples, 18, 8, 8), dtype=np.uint8)
        targets = np.zeros(num_samples, dtype=np.float32)
        
        fen_list = df['FEN'].to_list()
        eval_list = df['Evaluation'].to_list()

        pbar = tqdm.tqdm(total=num_samples, desc="Transforming dataset", unit="positions")

        for i, (fen, evaluation) in enumerate(zip(fen_list, eval_list)):
            features[i] = fen_to_tensor(fen)
            
            # 3. Store normalized evaluation
            targets[i] = evaluation
            
            if i % 100000 == 0:
                pbar.update(100000)
        
        logger.info("Converting to PyTorch tensors.")

        features_tensor = torch.from_numpy(features)
        targets_tensor = torch.from_numpy(targets)

        del features, targets

        logger.info(f"Saving to {output_filepath}.")

        torch.save({
            'features_tensor': features_tensor,
            'targets_tensor': targets_tensor,
            'num_samples': num_samples,
        }, output_filepath)