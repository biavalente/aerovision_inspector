from __future__ import annotations

import torch
from torch import nn


CLASS_NAMES = ("good", "ablation", "breakdown", "fracture", "groove")


class BladeDefectClassifier(nn.Module):
    """Compact classifier kept deliberately small for offline CPU demonstration."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.15), nn.Linear(64, num_classes))

    def forward_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image)
        logits = self.classifier(self.pool(features).flatten(1))
        return logits, features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _features = self.forward_features(image)
        return logits
