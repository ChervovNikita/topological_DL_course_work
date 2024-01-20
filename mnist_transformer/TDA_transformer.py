from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
import gudhi as gd
import math
from typing import Text
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
import sys


def diagram(image, sublevel=True):
    # get height and square image
    h = int(np.sqrt(image.shape[0]))
    image_sq = image.reshape((h,h))

    # create complex
    cmplx = gd.CubicalComplex(dimensions=(h, h), top_dimensional_cells=(sublevel*image))

    # get pairs of critical simplices
    cmplx.compute_persistence()
    critical_pairs = cmplx.cofaces_of_persistence_pairs()
    
    # get essential critical pixel
    bpx0_essential = critical_pairs[1][0][0] // h, critical_pairs[1][0][0] % h

    # get critical pixels corresponding to critical simplices
    try:
        bpx0 = [critical_pairs[0][0][i][0] for i in range(len(critical_pairs[0][0]))]
        dpx0 = [critical_pairs[0][0][i][1] for i in range(len(critical_pairs[0][0]))]
    except IndexError:
        bpx0 = [[]]
        dpx0 = [[]]
        
    try:
        bpx1 = [critical_pairs[0][1][i][0] for i in range(len(critical_pairs[0][1]))]
        dpx1 = [critical_pairs[0][1][i][1] for i in range(len(critical_pairs[0][1]))]
    except IndexError:
        bpx1 = [[]]
        dpx1 = [[]]
    

    flat_image = image_sq.flatten()
    pd0_essential = torch.tensor([[image_sq[bpx0_essential], torch.max(image)]])

    if (len(bpx0)!=0):
        pdb0 = flat_image[bpx0][:, None]
        pdd0 = flat_image[dpx0][:, None]
        pd0 = torch.Tensor(np.hstack([pdb0, pdd0]))
        pd0 = torch.vstack([pd0, pd0_essential])
    else:
        pd0 = pd0_essential

    if (len(bpx1)!=0):
        pdb1 = flat_image[bpx1][:, None]
        pdd1 = flat_image[dpx1][:, None]
        pd1 = torch.Tensor(np.hstack([pdb1, pdd1]))
    else:
        pd1 = torch.zeros((1, 2))
    
    return pd0, pd1


def process_by_direction(img, alpha):
    X = (math.cos(alpha) - (np.arange(0, img.shape[0]) - (img.shape[0] / 2 - 0.5)) / (img.shape[0] * math.sqrt(2))) * math.cos(alpha) / 2
    Y = (math.sin(alpha) - (np.arange(0, img.shape[1]) - (img.shape[1] / 2 - 0.5)) / (img.shape[1] * math.sqrt(2))) * math.sin(alpha) / 2
    direction_filter = X.reshape(-1, 1) + Y.reshape(1, -1)
    return np.maximum(direction_filter, img)


def process_image(img, filter_params):
    imgs = [process_by_direction(img.reshape(28, 28), alpha) for alpha in filter_params]
    diagrams = []
    for i, img in enumerate(imgs):
        res = diagram(torch.Tensor(img.flatten()))
        for j in range(len(res)):
            diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, filter_params[i]] for _ in range(res[j].shape[0])])], axis=1))

    diagrams = torch.concatenate(diagrams)
    return diagrams


def process_by_conv(img, conv):
    img = conv(torch.Tensor(img).reshape(1, 28, 28)).detach()
    diagrams = []
    for i in range(img.shape[0]):
        res = diagram(img[i].flatten())
        for j in range(len(res)):
            diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, i] for _ in range(res[j].shape[0])])], axis=1))
    diagrams = torch.concatenate(diagrams)
    return diagrams


class MyDatasetDirection(torch.utils.data.Dataset):
    def __init__(self, X, y, filter_params):
        self.X = [
            process_image(x / 255, filter_params).numpy() for x in tqdm(X)
        ]
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyDatasetConv(torch.utils.data.Dataset):
    def __init__(self, X, y, conv):
        with torch.no_grad():
            self.X = [
                process_by_conv(x / 255, conv).detach().numpy() for x in tqdm(X)
            ]
            self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
    for el in train_dataset:
        max_len = max(max_len, el[0].shape[0])
    seq_size = 1
    while seq_size < max_len * 1.2:
        seq_size *= 2
    
    logger = TensorBoardLogger("tb_logs", name=model_name)
    data = DataModule(train_path, test_path)
    model = Transformer(n_in=4, n_hidden=8, n_out=10, seq_size=seq_size)
    trainer = Trainer(accelerator="gpu", devices=1, min_epochs=100, max_epochs=300, logger=logger)
    trainer.fit(model=model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())


if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_name = sys.argv[3]
    train(train_path, test_path, model_name)
