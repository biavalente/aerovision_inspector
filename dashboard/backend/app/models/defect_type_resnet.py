from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


# These are AeBAD-S source labels. They are deliberately kept separate from a
# future operational aircraft-maintenance taxonomy.
DEFECT_TYPE_CLASSES = ("ablation", "breakdown", "fracture", "groove")
# A normal ROI is a rejection class for false residual proposals. It is not a
# defect type and is never surfaced as a final defect classification.
ROI_CLASS_NAMES = ("normal_roi", *DEFECT_TYPE_CLASSES)


class BasicBlock(nn.Module):
    """The standard ResNet-18 residual block, implemented without torchvision."""

    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        identity = image
        output = self.relu(self.bn1(self.conv1(image)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(image)
        return self.relu(output + identity)


class DefectTypeResNet18(nn.Module):
    """ImageNet-initialised ResNet-18 adapted to a local defect-type task.

    The module exposes the final convolutional activations so the inspection
    service can produce Grad-CAM evidence for the human reviewer.
    """

    def __init__(self, num_classes: int = len(ROI_CLASS_NAMES)) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(self.in_channels, channels, stride=stride)]
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.relu(self.bn1(self.conv1(image)))
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        activations = self.layer4(output)
        logits = self.fc(self.avgpool(activations).flatten(1))
        return logits, activations

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _activations = self.forward_features(image)
        return logits

    def load_backbone_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load any compatible ResNet-18 backbone while intentionally replacing fc."""

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("Os pesos ResNet-18 não contêm um state_dict válido.")
        backbone_state = {key: value for key, value in state_dict.items() if not key.startswith("fc.")}
        incompatible = self.load_state_dict(backbone_state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("fc.")]
        missing = [key for key in incompatible.missing_keys if not key.startswith("fc.")]
        if unexpected or missing:
            raise RuntimeError(
                "Os pesos não são compatíveis com a arquitetura ResNet-18 local: "
                f"unexpected={unexpected}, missing={missing}"
            )

    def load_imagenet_backbone(self, checkpoint_path: str | Path) -> None:
        """Compatibility alias for the original ImageNet initialisation path."""
        self.load_backbone_checkpoint(checkpoint_path)
