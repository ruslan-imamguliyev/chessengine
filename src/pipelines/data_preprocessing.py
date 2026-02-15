from src.config.manager import ConfigurationManager
from src.data.preprocessing import DataPreprocessor
from src.logging import logger


if __name__ == '__main__':
    logger.info("Initiating Data Preprocessing pipeline.")

    try:
        config = ConfigurationManager()
        preprocessor = DataPreprocessor(config)
        preprocessor.preprocess()
    except Exception as e:
        logger.exception("Data Preprocessing pipeline failed: " + str(e))
        raise e
    
    logger.info("Data Preprocessing pipeline completed.")