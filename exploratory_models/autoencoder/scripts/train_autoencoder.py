from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATASETS_DIR, MODELS_DIR  # noqa: E402
from backend.app.db import connect, init_db, now_iso  # noqa: E402
from backend.app.models.autoencoder import BladeAutoencoder  # noqa: E402
from backend.app.services.image_analysis import MODEL_INPUT_SIZE  # noqa: E402


class NormalBladeDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Imagem inacessível: {self.paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.0)


def top_error(model: nn.Module, image: torch.Tensor) -> float:
    with torch.inference_mode():
        reconstruction = model(image.unsqueeze(0)).squeeze(0)
    error = torch.mean(torch.abs(image - reconstruction), dim=0).numpy()
    return float(np.percentile(error, 99.2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino local de autoencoder em imagens normais AeBAD-S.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0, help="0 usa todas as imagens normais disponíveis.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    train_root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "train" / "good"
    paths = sorted(
        path
        for path in train_root.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if args.max_images:
        paths = paths[: args.max_images]
    if len(paths) < 8:
        raise RuntimeError(f"São necessárias imagens normais AeBAD-S em {train_root}.")

    dataset = NormalBladeDataset(paths)
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=True, num_workers=0)
    model = BladeAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        for batch in loader:
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch {epoch}/{args.epochs}: l1={np.mean(losses):.6f}", flush=True)

    model.eval()
    calibration_scores = [top_error(model, dataset[index]) for index in range(len(dataset))]
    calibration = {
        "top_error_p50": float(np.percentile(calibration_scores, 50)),
        "top_error_p95": float(np.percentile(calibration_scores, 95)),
        "top_error_p99": float(np.percentile(calibration_scores, 99)),
    }
    version_name = f"blade_autoencoder_aebad_s_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, MODELS_DIR / "blade_autoencoder.pt")
    metadata = {
        "version_name": version_name,
        "model_type": "convolutional_autoencoder",
        "trained_at": now_iso(),
        "input_size": MODEL_INPUT_SIZE,
        "training_dataset": "AeBAD-S/train/good",
        "training_image_count": len(paths),
        "epochs": args.epochs,
        "loss": "L1 reconstruction loss",
        "calibration": calibration,
        "limitations": "Treinado apenas em imagens normais AeBAD-S; usar como apoio à revisão humana.",
    }
    (MODELS_DIR / "blade_autoencoder.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    init_db()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                version_name, model_type, status, created_at, dataset_snapshot_json, metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_name) DO UPDATE SET status=excluded.status, metrics_json=excluded.metrics_json
            """,
            (
                version_name,
                "convolutional_autoencoder",
                "active",
                now_iso(),
                json.dumps({"dataset": "AeBAD-S/train/good", "images": len(paths)}, ensure_ascii=False),
                json.dumps({"training_l1": float(np.mean(losses)), **calibration}, ensure_ascii=False),
                "Baseline local de anomalia para demonstração; não é um modelo de certificação.",
            ),
        )
        connection.commit()

    print(json.dumps({"version": version_name, "images": len(paths), "calibration": calibration}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
