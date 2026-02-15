from src.config.manager import ConfigurationManager
from src.data.transformation import DataTransformator
from src.logging import logger


if __name__ == '__main__':
    logger.info("Initiating Data Transformation pipeline.")

    try:
        config = ConfigurationManager()
        transformator = DataTransformator(config)
        transformator.transform()
    except Exception as e:
        logger.exception("Data Transformation pipeline failed: " + str(e))
        raise e
    
    logger.info("Data Transformation pipeline completed.")