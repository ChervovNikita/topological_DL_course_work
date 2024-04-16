from numpy import diag
import torch
from typing import Text
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.nn.functional as F


class DiagramsDataset(torch.utils.data.Dataset):
    def __init__(self, diagrams, labels):
        self.diagrams = diagrams
        self.labels = labels
        
    def __len__(self):
        return len(self.diagrams)
    
    def __getitem__(self, idx):
        return self.diagrams[idx], self.labels[idx]


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_collate_fn(seq_size: int):
    def collate_fn(batch):
        diagrams, labels = zip(*batch)
        diagrams = [
            diagram[:seq_size] for diagram in diagrams
        ]
        diagrams = [
            F.pad(diagram, (0, 0, seq_size - diagram.shape[0], 0)) for diagram in diagrams
        ]
        labels = torch.LongTensor(labels)
        diagrams = torch.stack(diagrams, dim=0)
        return diagrams, labels
    return collate_fn


class DataModuleDiagrams(LightningDataModule):
    def __init__(self, train_path: Text, test_path: Text, seq_size: int, batch_size: int = 4):
        super().__init__()
        self.train_dataset = torch.load(train_path)
        self.test_dataset = torch.load(test_path)
        self.batch_size = batch_size
        self.seq_size = seq_size

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=get_collate_fn(self.seq_size))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=get_collate_fn(self.seq_size))


class DataModuleImages(LightningDataModule):
    def __init__(self, train_path: Text, test_path: Text, batch_size: int = 4):
        super().__init__()
        self.train_dataset = torch.load(train_path)
        self.test_dataset = torch.load(test_path)
        self.batch_size = batch_size

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
