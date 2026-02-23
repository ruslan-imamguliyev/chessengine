import modal

# Create Modal app
app = modal.App("chess-training")

image = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image, gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 3
)
def train():
    import sys
    sys.path.append("/root")

    from src.config.manager import ConfigurationManager
    from src.data.entity import DatasetEntity
    from src.models import ModelManager
    from src.models.trainer import ChessTrainerMLflow
    from src.logging import logger


    if __name__ == '__main__':
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
