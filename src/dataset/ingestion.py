import os
from requests.exceptions import ConnectionError
from time import sleep
import shutil
import kagglehub
import zipfile
from src.logging import logger
from abc import ABC, abstractmethod
from pathlib import Path
from box import ConfigBox
from src.config.manager import ConfigurationManager


class DatasetIngestor(ABC):
    @abstractmethod
    def ingest(self) -> None:
        pass


class KaggleDatasetIngestor(DatasetIngestor):
    def __init__(
            self,
            config: ConfigBox
            ):
        self.config = config

    def ingest(self) -> None:
        current_path = Path(os.getcwd()) / Path(self.config.install_path)
        if not os.path.exists(current_path):
            logger.info("Making directory for the dataset: " + str(current_path))
            os.mkdir(current_path)
        
        logger.info("Downloading the dataset to this path: " + str(current_path))
        
        try:
            path_str = kagglehub.dataset_download(self.config.source)
        except ConnectionError as e:
            logger.exception(e)
            logger.info("Trying to request the download one more time in 3 seconds.")
            sleep(3)
            self.ingest()
            return
        
        downloaded = Path(path_str)
        
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
        method = self.config.method
        available_methods = self.config.available_methods
        
        if not method in available_methods:
            raise ValueError("No such data ingestion method: " + self.config.method)

        config = ConfigBox(available_methods[method])

        match self.config.method:
            case "kagglehub":
                return KaggleDatasetIngestor(config)