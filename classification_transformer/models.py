# all imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import diagram


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
