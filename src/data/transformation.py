import numpy as np
from src.utils import fen_to_tensor
from src.logging import logger
from math import ceil
import os
from src.config.manager import ConfigurationManager
import pandas as pd


class DataTransformator:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_data_transformation_config()


    def transform(self) -> None:

        if not os.path.exists(self.config.output_path):
            logger.info("Making directory for the transformed dataset: " + self.config.output_path)
            try:
                os.makedirs(self.config.output_path)
            except Exception as e:
                logger.exception("Failed to create directory for the transformed dataset.")
                raise e
        
        logger.info("Reading the full dataset.")
        df = pd.read_csv(self.config.input_file)


        num_samples = len(df)

        logger.info("Initializing memory-mapped files for features and targets.")

        try:
            feature_filepath = os.path.join(
                self.config.output_path,
                self.config.feature_filename
            )
            target_filepath = os.path.join(
                self.config.output_path,
                self.config.target_filename
            )

            X = np.memmap(
                f"{feature_filepath}_x.bin",
                dtype='uint8', mode='w+',
                shape=(num_samples, 18, 8)
            )

            Y = np.memmap(
                f"{target_filepath}_y.bin",
                dtype='float32',
                mode='w+',
                shape=(num_samples,)
            )
        except Exception as e:
            logger.exception("Failed to initialize memory-mapped files.")
            raise e
        
        logger.info("Processing dataset in chunks and writing to memory-mapped files.")
        
        fen_list = df['FEN'].to_list()
        eval_list = df['Evaluation'].to_list()

        for i, (fen, evaluation) in enumerate(zip(fen_list, eval_list)):
            tensor_bool = fen_to_tensor(fen)

            X[i] = np.packbits(tensor_bool, axis=-1).reshape(18, 8)
            
            # 3. Store normalized evaluation
            Y[i] = evaluation
            
            if i % 100000 == 0:
                X.flush() # Periodically write to disk to keep RAM clear
                print(f"Processed {i} positions...")
        
        del X, Y

        logger.info("Successfully transformed the dataset and saved to memory-mapped files.")

