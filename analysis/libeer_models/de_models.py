from __future__ import annotations

import torch
from torch import nn

from analysis.libeer_models.common import ClassifierFromExtractor, DEFeatureExtractor


def build_de_model(name: str, *, n_channels: int, n_bands: int, n_classes: int):
    builders = {
        "DGCNN": lambda: GraphClassifier(n_channels, n_bands, n_classes, hidden_dim=128),
        "GCBNet": lambda: GCBNet(n_channels, n_bands, n_classes),
        "GCBNet_BLS": lambda: GCBNetBLS(n_channels, n_bands, n_classes),
        "CDCN": lambda: CDCN(n_channels, n_bands, n_classes),
        "DBN": lambda: DBN(n_channels, n_bands, n_classes),
        "HSLT": lambda: HSLT(n_channels, n_bands, n_classes),
        "RGNN": lambda: GraphClassifier(n_channels, n_bands, n_classes, hidden_dim=256),
        "RGNN_official": lambda: GraphClassifier(n_channels, n_bands, n_classes, hidden_dim=256),
    }
    if name not in builders:
        raise ValueError(f"Unknown DE model: {name}")
    return builders[name]()


class GraphClassifier(ClassifierFromExtractor):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int, hidden_dim: int = 128):
        super().__init__(DEFeatureExtractor(n_channels, n_bands, hidden_dim=hidden_dim, graph=True), n_classes)


class GCBNet(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int):
        super().__init__()
        self.graph = DEFeatureExtractor(n_channels, n_bands, graph=True)
        self.gate = nn.Sequential(nn.Linear(self.graph.output_dim, self.graph.output_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(self.graph.output_dim, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        features = self.graph(x)
        return self.classifier(features * self.gate(features))


class GCBNetBLS(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int):
        super().__init__()
        self.graph = DEFeatureExtractor(n_channels, n_bands, graph=True)
        self.broad = nn.ModuleList([nn.Linear(self.graph.output_dim, 32) for _ in range(4)])
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        features = self.graph(x)
        broad = torch.cat([torch.tanh(layer(features)) for layer in self.broad], dim=1)
        return self.classifier(broad)


class CDCN(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, n_channels, n_bands)),
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class DBN(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        input_dim = n_channels * n_bands
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden1), nn.Sigmoid(), nn.Linear(hidden1, hidden2), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(hidden2, hidden1), nn.Sigmoid(), nn.Linear(hidden1, input_dim))
        self.classifier = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x.reshape(x.shape[0], -1)))

    def reconstruction_loss(self, x):
        flat = x.reshape(x.shape[0], -1)
        return torch.nn.functional.mse_loss(self.decoder(self.encoder(flat)), flat)


class HSLT(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int, model_dim: int = 64):
        super().__init__()
        self.embedding = nn.Linear(n_bands, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.3,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, n_classes))

    def forward(self, x):
        x = self.encoder(self.embedding(x))
        return self.classifier(x.mean(dim=1))
