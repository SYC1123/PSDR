import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, in_dim, proj_dim=128, hidden_dim=0):
        super().__init__()
        if proj_dim <= 0:
            raise ValueError("proj_dim must be > 0")
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, proj_dim),
            )
        else:
            self.net = nn.Linear(in_dim, proj_dim)

    def forward(self, x):
        return self.net(x)


class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, scale=16.0, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        weight = F.normalize(self.weight, p=2, dim=1, eps=self.eps)
        return torch.matmul(x, weight.t()) * self.scale
