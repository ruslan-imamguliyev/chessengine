import os
import shutil
import kagglehub
import zipfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from src.config import ConfigurationManager, DataIngestionConfig


class DatasetIngestor(ABC):
    @abstractmethod
    def ingest(self) -> None:
        pass


class KaggleDatasetIngestor(DatasetIngestor):
    def __init__(
            self,
            config: DataIngestionConfig
            ):
        self.config = config

    def ingest(self) -> None:
        path_str = kagglehub.dataset_download(self.config.source)
        downloaded = Path(path_str)
        current_path = Path(os.getcwd()) / Path(self.config.install_path)
        if not downloaded.exists():
            raise FileNotFoundError(f"Downloaded path does not exist: {downloaded}")

        try:
            if downloaded.is_file() and downloaded.suffix == ".zip":
                with zipfile.ZipFile(downloaded, "r") as zf:
                    zf.extractall(current_path)
                try:
                    downloaded.unlink()
                except Exception as e:
                    raise e
                    #self.logger.debug("Could not remove downloaded zip", exc_info=True)

            elif downloaded.is_dir():
                for item in downloaded.iterdir():
                    dest = current_path / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(current_path))
                try:
                    downloaded.rmdir()
                except OSError as e:
                    raise e
                    #self.logger.debug("Could not remove download directory (not empty)", exc_info=True)

            else:
                raise RuntimeError(f"Unsupported downloaded artifact: {downloaded}")

        except Exception as e:
            #self.logger.exception("Failed to ingest dataset")
            raise e


class DataIngestionManager:
    def __init__(
            self,
            config: ConfigurationManager
        ):
        self.config = config.get_data_ingestion_config()
    
    def get_data_ingestor(self) -> DatasetIngestor:
        match self.config.type:
            case "kagglehub":
                return KaggleDatasetIngestor(self.config)