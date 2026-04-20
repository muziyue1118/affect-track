from __future__ import annotations

import torch
from torch import nn


def build_raw_model(name: str, *, n_channels: int, n_classes: int, n_samples: int):
    builders = {
        "shallow_convnet": lambda: ShallowConvNet(n_channels, n_classes, n_samples),
        "EEGNet": lambda: EEGNet(n_channels, n_classes, n_samples),
        "TSception": lambda: TSception(n_channels, n_classes, n_samples),
        "ACRNN": lambda: ACRNN(n_channels, n_classes, n_samples),
        "FBSTCNet": lambda: FBSTCNet(n_channels, n_classes, n_samples),
    }
    if name not in builders:
        raise ValueError(f"Unknown raw EEG model: {name}")
    return builders[name]()


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_samples: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, n_channels, n_samples)),
            nn.Conv2d(1, 16, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_samples: int, f1: int = 8, depth: int = 2):
        super().__init__()
        f2 = f1 * depth
        self.features = nn.Sequential(
            nn.Unflatten(1, (1, n_channels, n_samples)),
            nn.Conv2d(1, f1, kernel_size=(1, max(8, n_samples // 8)), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(f2, f2, kernel_size=(1, 16), padding="same", groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(f2, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x.reshape(x.shape[0], -1)))


class TSception(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_samples: int, num_t: int = 12, num_s: int = 12):
        super().__init__()
        windows = [0.5, 0.25, 0.125]
        self.temporal = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, num_t, kernel_size=(1, max(3, int(n_samples * ratio))), padding="same"),
                    nn.BatchNorm2d(num_t),
                    nn.ELU(),
                    nn.AvgPool2d((1, 4)),
                )
                for ratio in windows
            ]
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(num_t * len(windows), num_s, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(num_s),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_s, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        temporal = torch.cat([block(x) for block in self.temporal], dim=1)
        return self.spatial(temporal)


class ACRNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_samples: int, hidden_dim: int = 64):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(4),
        )
        self.rnn = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden_dim * 2, n_classes))

    def forward(self, x):
        weights = self.channel_attention(x).unsqueeze(-1)
        x = self.conv(x * weights)
        x = x.transpose(1, 2)
        output, _ = self.rnn(x)
        return self.classifier(output[:, -1])


class FBSTCNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_samples: int):
        super().__init__()
        self.filterbank = nn.Sequential(
            nn.Conv1d(n_channels, n_channels * 4, kernel_size=25, padding=12, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels * 4),
            nn.ELU(),
        )
        self.mixer = nn.Sequential(
            nn.Conv1d(n_channels * 4, 64, kernel_size=1),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.mixer(self.filterbank(x))
