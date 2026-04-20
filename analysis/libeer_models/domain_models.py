from __future__ import annotations

import torch
from torch import nn

from analysis.libeer_models.common import DEFeatureExtractor, DomainClassifier, MLPHead


def build_source_generalization_model(
    name: str,
    *,
    n_channels: int,
    n_bands: int,
    n_classes: int,
    num_domains: int,
):
    if name in {"CoralDgcnn", "DannDgcnn"}:
        return SourceGeneralizationNet(n_channels, n_bands, n_classes, num_domains)
    raise ValueError(f"Unknown source-domain generalization model: {name}")


def build_domain_adaptation_model(
    name: str,
    *,
    input_kind: str,
    n_channels: int,
    n_bands: int,
    n_classes: int,
    num_domains: int,
):
    if name in {"MsMDA", "NSAL_DGAT", "PRRL"}:
        graph = name == "NSAL_DGAT"
        return TensorDomainAdaptationNet(n_channels, n_bands, n_classes, graph=graph)
    raise ValueError(f"Unknown domain adaptation model: {name}")


class SourceGeneralizationNet(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int, num_domains: int):
        super().__init__()
        self.extractor = DEFeatureExtractor(n_channels, n_bands, graph=True)
        self.classifier = MLPHead(self.extractor.output_dim, n_classes)
        self.domain_classifier = DomainClassifier(self.extractor.output_dim, num_domains=max(2, num_domains))

    def forward(self, x, alpha: float = 1.0):
        features = self.extractor(x)
        return {
            "logits": self.classifier(features),
            "domain_logits": self.domain_classifier(features, alpha=alpha),
        }

    def predict_logits(self, x):
        return self.classifier(self.extractor(x))


class TensorDomainAdaptationNet(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, n_classes: int, graph: bool = False):
        super().__init__()
        self.extractor = DEFeatureExtractor(n_channels, n_bands, graph=graph)
        self.classifier = MLPHead(self.extractor.output_dim, n_classes)
        self.domain_classifier = DomainClassifier(self.extractor.output_dim, num_domains=2)

    def forward(self, source, target=None, alpha: float = 1.0):
        source_features = self.extractor(source)
        source_logits = self.classifier(source_features)
        if target is None:
            return source_logits
        target_features = self.extractor(target)
        domain_features = torch.cat([source_features, target_features], dim=0)
        return {
            "logits": source_logits,
            "target_logits": self.classifier(target_features),
            "domain_logits": self.domain_classifier(domain_features, alpha=alpha),
        }

    def predict_logits(self, x):
        return self.classifier(self.extractor(x))
