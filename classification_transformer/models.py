# all imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import diagram, distance_transform_edt
import math
import numpy as np


class ConvDiagram(nn.Module):
    def __init__(self, device):
        super(ConvDiagram, self).__init__()
        self.device = device
        
    def forward(self, x):
        diagrams = []
        for i in range(x.shape[0]):
            res = diagram(x[i].flatten(), self.device)
            for j in range(len(res)):
                diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, i] for _ in range(res[j].shape[0])]).to(self.device)], axis=1))
        diagrams = torch.concatenate(diagrams)
        return diagrams


class Transformer(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, seq_size=1024, nhead=2, num_layers=2, dim_feedforward=16):
        super(Transformer, self).__init__()
        self.embeddings = nn.Linear(n_in, n_hidden)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(seq_size, n_out)

    def forward(self, X):
        X = self.embeddings(X)
        X = self.transformer(X)
        X = X.mean(dim=-1)
        X = self.classifier(X)
        X = X.softmax(dim=-1)
        return X


class TopologicalConvTransformer(nn.Module):
    def __init__(self, n_in, n_conv, max_sequence, n_diag, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16, device='cuda'):
        super(TopologicalConvTransformer, self).__init__()
        
        self.max_sequence = max_sequence
        self.conv = nn.Conv2d(n_in, n_conv, 3)
        self.diagram = ConvDiagram(device)
        self.transformer = Transformer(n_diag, n_hidden, n_out, max_sequence, nhead, num_layers, dim_feedforward)

    def forward(self, xs):
        result = []
        for i in range(xs.shape[0]):
            x = xs[i][None, :, :] / 256
            x = self.conv(x)
            x = self.diagram(x)
            if x.shape[0] > self.max_sequence:
                x = x[:self.max_sequence]
            x = F.pad(x, (0, 0, 0, self.max_sequence - x.shape[0]), "constant", 0)
            x = self.transformer(x)
            result.append(x[None, :])
        result = torch.concatenate(result, axis=0)
        return result


class BaselineDiagram(nn.Module):
    def __init__(self, device):
        super(BaselineDiagram, self).__init__()
        self.device = device
    
    def forward(self, x):
        diagrams = []
        res = diagram(torch.Tensor(x.flatten()), self.device)
        for j in range(len(res)):
            if not res[j].shape[0]:
                diagrams.append(torch.zeros(0, 4))
            else:
                diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, 1] for _ in range(res[j].shape[0])])], axis=1))
        diagrams = torch.concatenate(diagrams)
        return diagrams


class TopologicalBaselineTransformer(nn.Module):
    def __init__(self, max_sequence, n_diag, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16, device='cuda'):
        super(TopologicalBaselineTransformer, self).__init__()
        
        self.max_sequence = max_sequence
        self.diagram = BaselineDiagram(device)
        self.transformer = Transformer(n_diag, n_hidden, n_out, max_sequence, nhead, num_layers, dim_feedforward)
    
    def forward(self, xs):
        result = []
        for i in range(xs.shape[0]):
            x = xs[i][None, :, :] / 256
            x = self.diagram(x)
            if x.shape[0] > self.max_sequence:
                x = x[:self.max_sequence]
            x = F.pad(x, (0, 0, 0, self.max_sequence - x.shape[0]), "constant", 0)
            x = self.transformer(x)
            result.append(x[None, :])
        result = torch.concatenate(result, axis=0)
        return result


class CedtDiagram(nn.Module):
    def __init__(self, device):
        super(CedtDiagram, self).__init__()
        self.device = device
    
    def forward_baseline(self, x):
        diagrams = []
        res = diagram(torch.Tensor(x.flatten()), self.device)
        for j in range(len(res)):
            if not res[j].shape[0]:
                diagrams.append(torch.zeros(0, 4))
            else:
                diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, 1] for _ in range(res[j].shape[0])])], axis=1))
        diagrams = torch.concatenate(diagrams)
        return diagrams

    def forward(self, x):
        edt = torch.Tensor(distance_transform_edt(x > 0.5).reshape(1, x.shape[1], x.shape[2]))
        cedt = edt * (x > 0.5) - edt * (x <= 0.5)
        return self.forward_baseline(cedt)


class TopologicalCedtTransformer(nn.Module):
    def __init__(self, max_sequence, n_diag, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16, device='cuda'):
        super(TopologicalCedtTransformer, self).__init__()
        
        self.max_sequence = max_sequence
        self.diagram = CedtDiagram(device)
        self.transformer = Transformer(n_diag, n_hidden, n_out, max_sequence, nhead, num_layers, dim_feedforward)
    
    def forward(self, xs):
        result = []
        for i in range(xs.shape[0]):
            x = xs[i][None, :, :] / 256
            x = self.diagram(x)
            if x.shape[0] > self.max_sequence:
                x = x[:self.max_sequence]
            x = F.pad(x, (0, 0, 0, self.max_sequence - x.shape[0]), "constant", 0)
            x = self.transformer(x)
            result.append(x[None, :])
        result = torch.concatenate(result, axis=0)
        return result


class DirectionalDiagram(nn.Module):
    def __init__(self, filter_count, device):
        super(DirectionalDiagram, self).__init__()
        self.device = device
        self.filters = nn.Parameter(torch.randn(filter_count) * 2 * math.pi)

    def process_by_direction(self, img, alpha):
        X = (math.cos(alpha) - (np.arange(0, img.shape[0]) - (img.shape[0] / 2 - 0.5)) / (img.shape[0] * math.sqrt(2))) * math.cos(alpha) / 2
        Y = (math.sin(alpha) - (np.arange(0, img.shape[1]) - (img.shape[1] / 2 - 0.5)) / (img.shape[1] * math.sqrt(2))) * math.sin(alpha) / 2
        direction_filter = X.reshape(-1, 1) + Y.reshape(1, -1)
        return torch.Tensor(direction_filter) + img

    def forward(self, x):
        w = int(np.sqrt(x.flatten().shape[0]))
        imgs = [self.process_by_direction(x.reshape(w, w), alpha) for alpha in self.filters]
        diagrams = []
        for i, img in enumerate(imgs):
            res = diagram(torch.Tensor(img.flatten()), self.device)
            for j in range(len(res)):
                if not res[j].shape[0]:
                    diagrams.append(torch.zeros(0, 4))
                else:
                    diagrams.append(torch.concatenate([res[j], torch.Tensor([[j, self.filters[i]] for _ in range(res[j].shape[0])])], axis=1))

        diagrams = torch.concatenate(diagrams)
        return diagrams

class TopologicalDirectionalTransformer(nn.Module):
    def __init__(self, filter_count, max_sequence, n_diag, n_hidden, n_out, nhead=2, num_layers=2, dim_feedforward=16, device='cuda'):
        super(TopologicalDirectionalTransformer, self).__init__()
        
        self.max_sequence = max_sequence
        self.diagram = DirectionalDiagram(filter_count, device)
        self.transformer = Transformer(n_diag, n_hidden, n_out, max_sequence, nhead, num_layers, dim_feedforward)
    
    def forward(self, xs):
        result = []
        for i in range(xs.shape[0]):
            x = xs[i][None, :, :] / 256
            x = self.diagram(x)
            if x.shape[0] > self.max_sequence:
                x = x[:self.max_sequence]
            x = F.pad(x, (0, 0, 0, self.max_sequence - x.shape[0]), "constant", 0)
            x = self.transformer(x)
            result.append(x[None, :])
        result = torch.concatenate(result, axis=0)
        return result
