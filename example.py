from src.config.manager import ConfigurationManager
from src.data.ingestion import DataIngestionManager
from src.data.transformation import DataTransformator
from src.data.preprocessing import DataPreprocessor

config = ConfigurationManager()

# dim = DataIngestionManager(config=config)
# data_ingestor = dim.get_data_ingestor()

# data_ingestor.ingest()

data_preprocessor = DataPreprocessor(config=config)
data_preprocessor.preprocess()
