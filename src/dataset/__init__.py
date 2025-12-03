import shutil
import kagglehub
import os
from pathlib import Path


class DatasetIngestion:
    def __init__(self):
        pass

    def ingest(self) -> None:
        path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")

        current_path = Path(__file__).resolve().parent

        for file in os.listdir(path):
            try:
                shutil.move(os.path.join(path, file), current_path)
            except Exception as e:
                print(e)