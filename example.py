from src.config.manager import ConfigurationManager
from src.dataset.ingestion import DataIngestionManager

config = ConfigurationManager()

dim = DataIngestionManager(config=config)
data_ingestor = dim.get_data_ingestor()

data_ingestor.ingest()