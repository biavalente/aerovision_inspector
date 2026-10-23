"""Probe the ROI type classifier with a multi-scale image grid.

This is a local development diagnostic.  It does not use masks at inference;
the supplied mask is used only to report a ground-truth reference for AeBAD-S
development samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATASETS_DIR  # noqa: E402
from backend.app.models.defect_type_resnet import ROI_CLASS_NAMES  # noqa: E402
from backend.app.services.image_analysis import (  # noqa: E402
    _classifier_tensor,
    _load_active_type_classifier,
)


def _starts(maximum: int, side: int, step: int) -> list[int]:
    values = list(range(0, max(1, maximum - side + 1), step))
    last = max(0, maximum - side)
    if not values or values[-1] != last:
        values.append(last)
    return sorted(set(values))


def _tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    tiles: list[tuple[int, int, int, int]] = []
    for fraction in (0.20, 0.30, 0.45, 0.65):
        side = max(224, int(round(min(width, height) * fraction)))
        step = max(1, side // 2)
        for y in _starts(height, side, step):
            for x in _starts(width, side, step):
                tiles.append((x, y, min(width, x + side), min(height, y + side)))
    return tiles


def _distribution(probabilities: torch.Tensor) -> dict[str, float]:
    return {
        label: round(float(probabilities[index]), 4)
        for index, label in enumerate(ROI_CLASS_NAMES)
    }


def _probe(image_path: Path) -> dict[str, object]:
    model, metadata = _load_active_type_classifier()
    if model is None:
        raise RuntimeError("Não existe classificador ROI activo para testar.")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Imagem inacessível: {image_path}")
    height, width = image.shape[:2]
    tiles = _tiles(width, height)
    input_size = int(metadata.get("input_size", 224))
    temperature = max(0.1, float(metadata.get("metrics", {}).get("temperature", 1.0)))
    tensors = [_classifier_tensor(image[y1:y2, x1:x2], input_size) for x1, y1, x2, y2 in tiles]
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), 32):
            logits = model(torch.cat(tensors[start : start + 32], dim=0))
            probabilities.extend(torch.softmax(logits / temperature, dim=1).cpu())
    candidates = []
    for tile, probability in zip(tiles, probabilities):
        index = int(probability.argmax().item())
        candidates.append(
            {
                "tile": tile,
                "top_class": ROI_CLASS_NAMES[index],
                "confidence": round(float(probability[index]), 4),
                "distribution": _distribution(probability),
            }
        )
    defect_candidates = [item for item in candidates if item["top_class"] != "normal_roi"]
    ordered = sorted(
        defect_candidates or candidates,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    return {
        "image": image_path.name,
        "tile_count": len(tiles),
        "top": ordered[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("good", "ablation", "breakdown", "fracture", "groove"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    directory = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "test" / args.label / "view"
    files = [path for path in sorted(directory.glob("*.png")) if not path.name.startswith("._")]
    print(json.dumps(_probe(files[args.index]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
