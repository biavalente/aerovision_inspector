from __future__ import annotations

import torch
from torch import nn


class ReviewClassifier(nn.Module):
    """Small local binary head for reviewed complex cases.

    It is intentionally separate from the active anomaly model: only a reviewed
    candidate that passes a later validation gate may become active.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(96, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image).flatten(1)
        return self.classifier(features).squeeze(1)
