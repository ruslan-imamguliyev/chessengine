from src.config.manager import ConfigurationManager
from src.data.ingestion import DataIngestionManager
from src.logging import logger


if __name__ == '__main__':
    logger.info("Initiating Data Intgestion pipeline.")

    try:
        config = ConfigurationManager()
        ingestor = DataIngestionManager(config).get_data_ingestor()
        ingestor.ingest()
    except Exception as e:
        logger.exception("Data Ingestion pipeline failed: " + str(e))
        raise e
    
    logger.info("Data Ingestion pipeline completed.")