from src.config import ConfigurationManager
from src.dataset import DataIngestionManager

config = ConfigurationManager(
    'src/config/config.yaml'
)
dim = DataIngestionManager(
    config=config
)

data_ingestor = dim.get_data_ingestor()
data_ingestor.ingest()