from __future__ import annotations

import torch
from torch import nn

from analysis.libeer_models.common import DomainClassifier, MLPHead


def build_sequence_model(
    name: str,
    *,
    n_channels: int,
    n_bands: int,
    n_classes: int,
    sequence_length: int,
    num_domains: int,
):
    if name == "STRNN":
        return STRNN(n_channels, n_bands, n_classes, sequence_length)
    if name == "BiDANN":
        return SequenceDomainAdaptationNet(n_channels, n_bands, n_classes, hidden_dim=128)
    if name == "R2GSTNN":
        return SequenceDomainAdaptationNet(n_channels, n_bands, n_classes, hidden_dim=160)
    raise ValueError(f"Unknown sequence model: {name}")


class STRNN(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int, sequence_length: int, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Linear(n_channels * n_bands, hidden_dim)
        self.temporal = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden_dim * 2, n_classes))

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.embedding(x)
        output, _ = self.temporal(x)
        return self.classifier(output[:, -1])


class SequenceDomainAdaptationNet(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Linear(n_channels * n_bands, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_dim = hidden_dim * 2
        self.classifier = MLPHead(self.output_dim, n_classes, hidden_dim=hidden_dim)
        self.domain_classifier = DomainClassifier(self.output_dim, num_domains=2, hidden_dim=hidden_dim)

    def extract_features(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.embedding(x)
        output, _ = self.temporal(x)
        return output[:, -1]

    def predict_logits(self, x):
        return self.classifier(self.extract_features(x))

    def forward(self, source, target=None, alpha: float = 1.0):
        source_features = self.extract_features(source)
        source_logits = self.classifier(source_features)
        if target is None:
            return source_logits
        target_features = self.extract_features(target)
        domain_features = torch.cat([source_features, target_features], dim=0)
        return {
            "logits": source_logits,
            "target_logits": self.classifier(target_features),
            "domain_logits": self.domain_classifier(domain_features, alpha=alpha),
        }
