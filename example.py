from src.config.manager import ConfigurationManager
from src.data.ingestion import DataIngestionManager
from src.data.transformation import DataTransformator

config = ConfigurationManager()

# dim = DataIngestionManager(config=config)
# data_ingestor = dim.get_data_ingestor()

# data_ingestor.ingest()

transformator = DataTransformator(config=config)
transformator.transform()