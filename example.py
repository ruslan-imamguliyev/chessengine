from src.config.manager import ConfigurationManager
from src.data.entity import DatasetEntity
from src.models import ModelManager
from src.models.trainer import ChessTrainerMLflow


if __name__ == '__main__':
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