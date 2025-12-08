import os
from requests.exceptions import ConnectionError
from time import sleep
import shutil
import kagglehub
import zipfile
from src.logging import logger
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
        try:
            path_str = kagglehub.dataset_download(self.config.source)
        except ConnectionError as e:
            logger.exception(e)
            logger.info("Trying to request the download one more time in 3 seconds.")
            sleep(3)
            self.ingest()
            return
        downloaded = Path(path_str)
        current_path = Path(os.getcwd()) / Path(self.config.install_path)
        if not downloaded.exists():
            logger.exception(FileNotFoundError(f"Downloaded path does not exist: {downloaded}"))
            raise

        try:
            if downloaded.is_file() and downloaded.suffix == ".zip":
                with zipfile.ZipFile(downloaded, "r") as zf:
                    zf.extractall(current_path)
                try:
                    downloaded.unlink()
                except Exception as e:
                    logger.error("Could not remove downloaded zip.")
                    raise e

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
                    logger.error("Could not remove download directory (not empty)", exc_info=True)
                    raise e

            else:
                logger.exception(RuntimeError(f"Unsupported downloaded artifact: {downloaded}"))
                raise

        except Exception as e:
            logger.exception("Failed to ingest dataset")
            raise e
        
        logger.info("Succesfully downloaded the dataset.")


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