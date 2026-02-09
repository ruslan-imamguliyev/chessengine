import torch
from typing import List
from src.utils import batch_fen_to_tensor
from src.logging import logger
from math import ceil
import os
from src.config.manager import ConfigurationManager
import pandas as pd


class DataTransformator:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_data_transformation_config()


    def transform(self) -> None:
        df = pd.read_csv(self.config.input_file)

        self.save_dataset_shards(
            fens=df.FEN.to_list(),
            evals=df.Evaluation.to_list()
        )


    def save_dataset_shards(
            self,
            fens: List[str],
            evals: List[int]
        ) -> None:

        output_dir = self.config.output_dir

        if not os.path.exists(output_dir):
            logger.info("Making directory for the transformed dataset: " + str(output_dir))
            os.makedirs(output_dir)

        shard_size = self.config.shard_size
        num_shards = ceil(len(fens) / shard_size)

        for i in range(num_shards):
            start = i * shard_size
            end = min((i + 1) * shard_size, len(fens))

            x = batch_fen_to_tensor(fens[start:end])
            y = torch.tensor(evals[start:end], dtype=torch.float32).unsqueeze(1)

            torch.save(
                {"x": x, "y": y},
                f"{output_dir}/shard_{i:04d}.pt"
            )

            logger.info(f"Saved shard {i}/{num_shards}")