import shutil
import kagglehub
import zipfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path


class DatasetIngestor(ABC):
    @abstractmethod
    def ingest(self) -> None:
        pass


class KaggleDatasetIngestor(DatasetIngestor):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def ingest(self) -> None:
        path_str = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
        downloaded = Path(path_str)
        current_path = Path(__file__).resolve().parent

        if not downloaded.exists():
            raise FileNotFoundError(f"Downloaded path does not exist: {downloaded}")

        try:
            if downloaded.is_file() and downloaded.suffix == ".zip":
                with zipfile.ZipFile(downloaded, "r") as zf:
                    zf.extractall(current_path)
                try:
                    downloaded.unlink()
                except Exception:
                    self.logger.debug("Could not remove downloaded zip", exc_info=True)

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
                except OSError:
                    self.logger.debug("Could not remove download directory (not empty)", exc_info=True)

            else:
                raise RuntimeError(f"Unsupported downloaded artifact: {downloaded}")

        except Exception:
            self.logger.exception("Failed to ingest dataset")
            raise


class DataIngestionManager:
    def __init__(self, config):
        pass
    
    def get_data_ingestor() -> DatasetIngestor:
        pass