import modal


app = modal.App("chess-training")

image = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file(".env", remote_path="/root/.env")
)

volume = modal.Volume.from_name(
    "chess-pipeline-storage",
    create_if_missing=True
)

@app.function(
    image=image, gpu="A100", volumes={"/root/data": volume},
    cpu=8, memory=32768, timeout=60 * 60 * 3
)
def train():
    from src.config.manager import ConfigurationManager
    from src.data.entity import DatasetEntity
    from src.models import ModelManager
    from src.models.trainer import ChessTrainerMLflow
    from src.logging import logger


    
    logger.info("Initiating Model Training pipeline.")

    try:
        config = ConfigurationManager()

        dataset_entity = DatasetEntity(config)
        train_loader = dataset_entity.get_data_loader("train")
        val_loader = dataset_entity.get_data_loader("val")
        mm = ModelManager(config)

        model_name, model = mm.get_model_name(), mm.get_model()
        trainer = ChessTrainerMLflow(
            config=config,
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader
        )
        trainer.train()
    except Exception as e:
        logger.exception("Model Training pipeline failed: " + str(e))
        raise e
    
    logger.info("Model Training pipeline completed.")


@app.function(image=image, volumes={"/root/data": volume})
def ingest():
    from src.config.manager import ConfigurationManager
    from src.data.ingestion import DataIngestionManager
    from src.logging import logger

    logger.info("Initiating Data Intgestion pipeline.")

    try:
        config = ConfigurationManager()
        ingestor = DataIngestionManager(config).get_data_ingestor()
        ingestor.ingest()
    except Exception as e:
        logger.exception("Data Ingestion pipeline failed: " + str(e))
        raise e
    
    logger.info("Data Ingestion pipeline completed.")
    volume.commit()


@app.function(image=image, volumes={"/root/data": volume})
def preprocess():
    from src.config.manager import ConfigurationManager
    from src.data.preprocessing import DataPreprocessor
    from src.logging import logger


    
    logger.info("Initiating Data Preprocessing pipeline.")

    try:
        config = ConfigurationManager()
        preprocessor = DataPreprocessor(config)
        preprocessor.preprocess()
    except Exception as e:
        logger.exception("Data Preprocessing pipeline failed: " + str(e))
        raise e
    
    logger.info("Data Preprocessing pipeline completed.")
    volume.commit()


@app.function(image=image, volumes={"/root/data": volume})
def transform(timeout=60*15):
    from src.config.manager import ConfigurationManager
    from src.data.transformation import DataTransformator
    from src.logging import logger


    
    logger.info("Initiating Data Transformation pipeline.")

    try:
        config = ConfigurationManager()
        transformator = DataTransformator(config)
        transformator.transform()
    except Exception as e:
        logger.exception("Data Transformation pipeline failed: " + str(e))
        raise e
    
    logger.info("Data Transformation pipeline completed.")
    volume.commit()


@app.function()
def pipeline():
    # ingest.remote()
    # preprocess.remote()
    transform.remote()
    train.remote()

@app.local_entrypoint()
def main():
    pipeline.remote()