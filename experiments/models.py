import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pytorch_lightning import LightningModule
import transforms


class Transformer(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16):
        super(Transformer, self).__init__()
        self.embeddings = nn.Linear(n_in, n_hidden)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(n_hidden, n_out)

    def forward(self, X):
        X = self.embeddings(X)
        X = self.transformer(X)
        X = X.mean(dim=1)
        X = self.classifier(X)
        X = X.softmax(dim=-1)
        return X


class LightningTransformer(LightningModule):
    def __init__(self):
        super(LightningTransformer, self).__init__()
    
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

### StaticTransformer

class StaticTransformer(LightningTransformer):
    def __init__(self, n_in, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16, device="cuda"):
        super(StaticTransformer, self).__init__()
        self.transformer = Transformer(n_in, n_hidden, n_out, nhead, num_layers, dim_feedforward)
        self._device = device
    
    def forward(self, X):
        return self.transformer(X)


### Conv Transformer

class ConvTransformer(LightningTransformer):
    def __init__(self, n_dims, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16, device="cuda"):
        super(ConvTransformer, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(n_dims[i], n_dims[i+1], kernel_size=3) for i in range(len(n_dims) - 1)
        ])
        self.transformer = Transformer(4, n_hidden, n_out, nhead, num_layers, dim_feedforward)
        self._device = device

    def forward(self, image_batch):  # img.shape = [batch_size, w, h]
        image_batch = image_batch.float()[:, None, :, :]
        for conv in self.convs:
            image_batch = conv(image_batch)
        max_diagram_size = 0
        results = []
        for i in range(image_batch.shape[0]):
            imgs = [transforms.diagram(image_batch[i, j].flatten(), self._device) for j in range(image_batch.shape[1])]
            diagram = []
            for k, img in enumerate(imgs):
                for j in range(len(img)):
                    if not img[j].shape[0]:
                        diagram.append(torch.zeros(0, 4).to(self._device))
                    else:
                        diagram.append(torch.concatenate([img[j], torch.Tensor([[j, k] for _ in range(img[j].shape[0])]).to(self._device)], axis=1))
            diagram = torch.concatenate(diagram)
            results.append(diagram)
            max_diagram_size = max(max_diagram_size, diagram.shape[0])
        diagrams = []
        for i in range(image_batch.shape[0]):
            diagram = results[i]
            if diagram.shape[0] > max_diagram_size:
                diagram = diagram[:max_diagram_size]
            if diagram.shape[0] < max_diagram_size:
                diagram = torch.concatenate([diagram, torch.zeros(max_diagram_size - diagram.shape[0], 4).to(self._device)])
            diagrams.append(diagram)
        diagrams = torch.stack(diagrams)
        return self.transformer(diagrams)


### Directional Transformer

class DirectionalTransformer(LightningTransformer):
    def __init__(self, n_in, dir_count, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16, device="cuda"):
        super(DirectionalTransformer, self).__init__()
        self.transformer = Transformer(n_in, n_hidden, n_out, nhead, num_layers, dim_feedforward)
        self.dirs = nn.Parameter(torch.randn(dir_count) * 2 * math.pi)
        self._device = device
    
    def forward(self, image_batch):
        image_batch = image_batch.float()
        max_diagram_size = 0
        results = []
        for image_id in range(image_batch.shape[0]):
            imgs = [transforms.process_by_direction(image_batch[image_id], self.dirs[i]) for i in range(len(self.dirs))]
            diagram = []
            for i, img in enumerate(imgs):
                res = transforms.diagram(torch.Tensor(img.flatten()), self._device)
                for j in range(len(res)):
                    if not res[j].shape[0]:
                        diagram.append(torch.zeros(0, 4).to(self._device))
                    else:
                        diagram.append(torch.concatenate([res[j], torch.Tensor([[j, self.dirs[i]] for _ in range(res[j].shape[0])]).to(self._device)]))
            diagram = torch.concatenate(diagram)
            results.append(diagram)
            max_diagram_size = max(max_diagram_size, diagram.shape[0])
        diagrams = []
        for image_id in range(image_batch.shape[0]):
            diagram = results[image_id]
            if diagram.shape[0] > max_diagram_size:
                diagram = diagram[:max_diagram_size]
            if diagram.shape[0] < max_diagram_size:
                diagram = torch.concatenate([diagram, torch.zeros(max_diagram_size - diagram.shape[0], 4).to(self._device)])
            diagrams.append(diagram)
        diagrams = torch.stack(diagrams)
        return self.transformer(diagrams.float())
