from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
from typing import Text
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from utils import *
from datasets import *
import loguru


def collate_fn_fixed(data, points_count):
    tmp_pd, _ = data[0]

    n_batch = len(data)
    n_features_pd = tmp_pd.shape[1]
    n_points_pd = points_count
    inputs_pd = np.zeros((n_batch, n_points_pd, n_features_pd), dtype=float)
    labels = np.zeros(len(data))

    for i, (pd, label) in enumerate(data):
        inputs_pd[i][:len(pd)] = pd
        labels[i] = label

    return torch.Tensor(inputs_pd), torch.Tensor(labels).long()


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
        self.collate_fn = lambda x: collate_fn_fixed(x, self.seq_size)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)


class Transformer(LightningModule):
    def __init__(self, n_in, n_hidden, n_out, seq_size):
        super(Transformer, self).__init__()
        self.embeddings = nn.Linear(n_in, n_hidden)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=2, dim_feedforward=16, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.classifier = nn.Linear(seq_size, n_out)
    
    def forward(self, X):
        X = self.embeddings(X)
        X = self.transformer(X)
        X = X.mean(dim=2)
        X = self.classifier(X)
        X = X.softmax(dim=-1)
        return X

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

    def _prepare_batch(self, batch):
        x, y = batch
        return x, y

    def _common_step(self, batch, batch_idx, stage: str):
        input, target = self._prepare_batch(batch)
        output = self(input)
        loss = F.cross_entropy(output, target)
        accuracy = (output.argmax(-1) == target).float().mean().item()
        self.log(f"{stage}_loss", loss, on_step=True)
        self.log(f"{stage}_accuracy", accuracy, on_step=True)
        return loss


def train(train_path, test_path, model_name):
    train_dataset = torch.load(train_path)
    max_len = 0
    classes = set()
    for el in train_dataset:
        max_len = max(max_len, el[0].shape[0])
        classes.add(el[1])
    seq_size = 1
    while seq_size < max_len * 1.2:
        seq_size *= 2
    
    
    loguru.logger.info(f"Using {len(classes)} classes")
    logger = TensorBoardLogger("tb_logs", name=model_name)
    data = DataModule(train_path, test_path)
    model = Transformer(n_in=4, n_hidden=8, n_out=len(classes), seq_size=seq_size)
    trainer = Trainer(accelerator="gpu", devices=1, min_epochs=100, max_epochs=300, logger=logger)
    trainer.fit(model=model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())


if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_name = sys.argv[3]
    train(train_path, test_path, model_name)
