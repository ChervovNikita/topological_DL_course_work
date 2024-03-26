import torch
from utils import *
from tqdm import tqdm

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


class MyDatasetBaseline(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = [
            process_baseline(x / 255).detach().numpy() for x in tqdm(X)
        ]
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyDatasetCEDT(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = [
            process_cedt(x / 255).detach().numpy() for x in tqdm(X)
        ]
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MNIST_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].reshape(28, 28), int(self.y[idx])
