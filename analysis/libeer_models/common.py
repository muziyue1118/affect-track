from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha: float = 1.0):
    return GradientReverse.apply(x, alpha)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TensorFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class SimpleGraphBlock(nn.Module):
    def __init__(self, n_channels: int, in_features: int, out_features: int):
        super().__init__()
        self.adj = nn.Parameter(torch.eye(n_channels))
        self.proj = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(n_channels)

    def forward(self, x):
        adj = torch.softmax(self.adj, dim=-1)
        x = torch.einsum("ij,bjf->bif", adj, x)
        x = self.proj(x)
        x = self.norm(x)
        return F.elu(x)


class DEFeatureExtractor(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, hidden_dim: int = 128, graph: bool = False):
        super().__init__()
        self.graph = graph
        if graph:
            self.block1 = SimpleGraphBlock(n_channels, n_bands, 32)
            self.block2 = SimpleGraphBlock(n_channels, 32, 32)
            self.output_dim = n_channels * 32
        else:
            self.net = TensorFeatureExtractor(n_channels * n_bands, hidden_dim=hidden_dim)
            self.output_dim = self.net.output_dim

    def forward(self, x):
        if not self.graph:
            return self.net(x)
        x = self.block1(x)
        x = self.block2(x)
        return x.reshape(x.shape[0], -1)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, num_domains: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features, alpha: float = 1.0):
        return self.net(grad_reverse(features, alpha))


class ClassifierFromExtractor(nn.Module):
    def __init__(self, extractor: nn.Module, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.extractor = extractor
        self.classifier = MLPHead(extractor.output_dim, n_classes, hidden_dim=hidden_dim)

    def forward(self, x):
        return self.classifier(self.extractor(x))
