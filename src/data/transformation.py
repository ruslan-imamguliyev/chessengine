import numpy as np
import tqdm
from src.utils import fen_to_tensor
from src.logging import logger
from math import ceil
import os
from src.utils import create_paths
from src.config.manager import ConfigurationManager
import pandas as pd


class DataTransformator:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_data_transformation_config()


    def transform(self) -> None:
        
        create_paths(self.config.output_path)
        
        logger.info("Reading the full dataset.")
        df = pd.read_csv(self.config.input_file)

        logger.info("Shuffling and splitting the dataset into train and test sets.")

        shuffled_indices = np.random.permutation(len(df))
        test_split = int(len(df) * self.config.test_split)
        train_df = df.iloc[shuffled_indices[test_split:]]
        test_df = df.iloc[shuffled_indices[:test_split]]

        logger.info("Initializing memory-mapped files for train set.")

        self.save(train_df, "train")
        
        logger.info("Initializing memory-mapped files for test set.")

        self.save(test_df, "test")

        logger.info("Successfully transformed the dataset and saved to memory-mapped files.")

    def save(
            self,
            df: pd.DataFrame,
            prefix: str
        ) -> None:
        num_samples = len(df)
        try:
            feature_filepath = os.path.join(
                self.config.output_path,
                prefix + "_" + self.config.feature_filename
            )
            target_filepath = os.path.join(
                self.config.output_path,
                prefix + "_" + self.config.target_filename
            )

            X = np.memmap(
                f"{feature_filepath}.bin",
                dtype='uint8', mode='w+',
                shape=(num_samples, 18, 8)
            )

            Y = np.memmap(
                f"{target_filepath}.bin",
                dtype='float32',
                mode='w+',
                shape=(num_samples,)
            )
        except Exception as e:
            logger.exception("Failed to initialize memory-mapped files: " + str(e))
            raise e
        
        logger.info("Processing dataset in chunks and writing to memory-mapped files.")
        
        fen_list = df['FEN'].to_list()
        eval_list = df['Evaluation'].to_list()

        pbar = tqdm.tqdm(total=num_samples, desc="Transforming dataset", unit="positions")

        for i, (fen, evaluation) in enumerate(zip(fen_list, eval_list)):
            tensor_bool = fen_to_tensor(fen)

            X[i] = np.packbits(tensor_bool, axis=-1).reshape(18, 8)
            
            # 3. Store normalized evaluation
            Y[i] = evaluation
            
            if i % 100000 == 0:
                X.flush() # Periodically write to disk to keep RAM clear
                pbar.update(100000)
        
        del X, Y