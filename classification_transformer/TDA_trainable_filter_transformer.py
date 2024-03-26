from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch
from typing import Text
from utils import *
from datasets import *
from models import TopologicalConvTransformer
import loguru
import sys
import json
from pytorch_lightning.loggers import TensorBoardLogger


class DataModule(LightningDataModule):
    def __init__(self, train_path: Text, test_path: Text, batch_size: int = 4):
        self.train_dataset = torch.load(train_path)
        self.test_dataset = torch.load(test_path)
        self.batch_size = batch_size
        max_len = 0
        for el in self.train_dataset:
            max_len = max(max_len, el[0].shape[0])
        self.seq_size = 1
        while self.seq_size < max_len * 1.2:
            self.seq_size *= 2

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class ModelConv(LightningModule):
    def __init__(self, n_in, n_conv, n_hidden, n_out, max_sequence, n_diag, nhead=2, num_layers=2, dim_feedforward=16, device='cuda'):
        super(ModelConv, self).__init__()
        self.model = TopologicalConvTransformer(n_in, n_conv, max_sequence, n_diag, n_hidden, n_out, nhead, num_layers, dim_feedforward, device)
        self._device = device

    def forward(self, x):
        return self.model(x)

    def _prepare_batch(self, batch):
        x, y = batch
        return x, y

    def _common_step(self, batch, batch_idx, stage: str):
        input, target = self._prepare_batch(batch)
        input = input.to(self._device)
        target = target.to(self._device)
        output = self(input)
        loss = F.cross_entropy(output, target)
        accuracy = (output.argmax(-1) == target).float().mean().item()
        self.log(f"{stage}_loss", loss, on_step=True)
        self.log(f"{stage}_accuracy", accuracy, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))


def train(train_path, test_path, model_name, model_type, **kwargs):
    loguru.logger.info(f"Run model")
    data = DataModule(train_path, test_path)
    logger = TensorBoardLogger("tb_logs", name=model_name)
    if model_type == "conv":
        model = ModelConv(**kwargs)
    trainer = Trainer(accelerator=kwargs['device'], devices=1, min_epochs=100, max_epochs=300, logger=logger)
    trainer.fit(model=model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())


if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_name = sys.argv[3]
    model_type = sys.argv[4]
    kwargs = json.loads(sys.argv[5])
    train(train_path, test_path, model_name, model_type, **kwargs)
