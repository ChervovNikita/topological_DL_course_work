from torch.utils.data import DataLoader
import loguru
import sys
import json
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import DataModuleDiagrams, DataModuleImages
import models
from pytorch_lightning import Trainer


def train(train_path, test_path, model_name, model_type, devices, epoches_count, **kwargs):
    loguru.logger.info(f"Run model")
    logger = TensorBoardLogger("tb_logs", name=model_name)

    if model_type == "static":
        data = DataModuleDiagrams(train_path, test_path, seq_size=kwargs['seq_size'])
        model = models.StaticTransformer(**kwargs)
    elif model_type == "conv":
        data = DataModuleImages(train_path, test_path)
        model = models.ConvTransformer(**kwargs)
    elif model_type == "directional":
        data = DataModuleImages(train_path, test_path)
        model = models.DirectionalTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trainer = Trainer(accelerator=kwargs['device'], devices=devices, min_epochs=epoches_count, max_epochs=epoches_count, logger=logger)
    trainer.fit(model=model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())


if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_name = sys.argv[3]
    model_type = sys.argv[4]
    devices = int(sys.argv[5])
    epoches_count = int(sys.argv[6])
    model_kwargs = json.loads(sys.argv[7])
    train(train_path, test_path, model_name, model_type, devices, epoches_count, **model_kwargs)
