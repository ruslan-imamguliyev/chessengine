import numpy as np
from src.config.manager import ConfigurationManager
import pandas as pd
from glob import glob
import os
from src.logging import logger


class DataPreprocessor:
    def __init__(
            self,
            config: ConfigurationManager
        ):
        self.config = config.get_data_preprocessing_config()

    def preprocess(self):
        logger.info("Combining all dataframes into one.")
        dfs = []
        
        for path in glob(self.config.input_path + "/*.csv"):
            logger.info("Reading dataframe at " + path)

            df = pd.read_csv(path)
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True).drop(columns=["Move"])

        logger.info("Normalizing `Evaluation` column.")
        max_eval = df[df["Evaluation"].str.contains("#")]["Evaluation"].str.split("#").str[1].astype(int).max()
        min_eval = df[df["Evaluation"].str.contains("#")]["Evaluation"].str.split("#").str[1].astype(int).min()

        df["Evaluation"] = df["Evaluation"].apply(
            lambda x: self.normalize(
                x,
                min_eval,
                max_eval
            ) if "#" in x else int(x)
        ).apply(lambda x: np.tanh(x / self.config.scaling_factor))

        
        if not os.path.exists(self.config.output_path):
            logger.info("Making directory for the preprocessed dataset: " + self.config.output_path)
            try:
                os.makedirs(self.config.output_path)
            except Exception as e:
                logger.exception("Failed to create directory for the preprocessed dataset.")
                raise e
            
        

        logger.info("Saving preprocessed dataframe to the output path: " + self.config.output_path)

        try:
            df.to_csv(os.path.join(
                self.config.output_path,
                self.config.output_file
            ), index=False)
        except Exception as e:
            logger.exception("Failed to save preprocessed dataframe.")
            raise e


    
    def normalize(
            self,
            x: str,
            minn: int, maxx: int
        ):
        num = int(x.split("#")[1])
        if num < 0:
            return -self.config.max_mate_value + \
                (self.config.max_mate_value - self.config.min_mate_value) * np.abs(num / minn)
        elif num > 0:
            return self.config.max_mate_value - \
                (self.config.max_mate_value - self.config.min_mate_value) * (num / maxx)
        return int(x.split("#")[1][0] + f"{self.config.max_mate_value}")