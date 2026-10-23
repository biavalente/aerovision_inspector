from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
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
from backend.app.models.blade_classifier import CLASS_NAMES, BladeDefectClassifier  # noqa: E402


INPUT_SIZE = 112


class StaticBladeDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Imagem inacessível: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
        if self.augment and random.random() < 0.45:
            alpha = random.uniform(0.88, 1.12)
            beta = random.uniform(-12, 12)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.0)
        return tensor, torch.tensor(target, dtype=torch.long)


def make_split(seed: int, max_per_class: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "test"
    generator = random.Random(seed)
    training: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    for class_index, label in enumerate(CLASS_NAMES):
        paths = sorted(
            path
            for path in (root / label).rglob("*")
            if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        generator.shuffle(paths)
        if max_per_class:
            paths = paths[:max_per_class]
        split_at = max(1, round(len(paths) * 0.8))
        training.extend((path, class_index) for path in paths[:split_at])
        validation.extend((path, class_index) for path in paths[split_at:])
    generator.shuffle(training)
    generator.shuffle(validation)
    return training, validation


def metrics_from_predictions(targets: list[int], predictions: list[int]) -> dict[str, float]:
    accuracy = sum(target == prediction for target, prediction in zip(targets, predictions)) / max(1, len(targets))
    result: dict[str, float] = {"validation_accuracy": round(accuracy, 3)}
    recalls: list[float] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        total = sum(target == class_index for target in targets)
        correct = sum(target == class_index and prediction == class_index for target, prediction in zip(targets, predictions))
        recall = correct / max(total, 1)
        result[f"recall_{class_name}"] = round(recall, 3)
        recalls.append(recall)
    result["balanced_recall"] = round(sum(recalls) / len(recalls), 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino local do classificador de desenvolvimento AeBAD-S.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=160, help="0 usa todas as imagens por classe.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    training, validation = make_split(args.seed, args.max_per_class)
    if not training or not validation:
        raise RuntimeError("Não foi possível construir o split de desenvolvimento AeBAD-S.")

    train_loader = DataLoader(StaticBladeDataset(training, augment=True), batch_size=max(1, args.batch_size), shuffle=True, num_workers=0)
    validation_loader = DataLoader(StaticBladeDataset(validation), batch_size=max(1, args.batch_size), shuffle=False, num_workers=0)
    class_counts = Counter(target for _, target in training)
    weights = torch.tensor([len(training) / max(class_counts[index], 1) for index in range(len(CLASS_NAMES))], dtype=torch.float32)
    model = BladeDefectClassifier()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets in train_loader:
            logits = model(images)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch {epoch}/{args.epochs}: cross_entropy={np.mean(losses):.5f}", flush=True)

    model.eval()
    targets_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for images, targets in validation_loader:
            predictions = model(images).argmax(dim=1)
            targets_all.extend(targets.tolist())
            predictions_all.extend(predictions.tolist())
    metrics = metrics_from_predictions(targets_all, predictions_all)
    version_name = f"blade_classifier_aebad_s_dev_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, MODELS_DIR / "blade_classifier.pt")
    metadata = {
        "version_name": version_name,
        "model_type": "supervised_static_blade_classifier",
        "trained_at": now_iso(),
        "input_size": INPUT_SIZE,
        "class_names": list(CLASS_NAMES),
        "training_dataset": "AeBAD-S/test development split",
        "training_image_count": len(training),
        "validation_image_count": len(validation),
        "epochs": args.epochs,
        "metrics": metrics,
        "limitations": (
            "Modelo de desenvolvimento treinado com um split interno de AeBAD-S/test. "
            "Não usar esta métrica como benchmark oficial nem como prova de generalização para outras aeronaves."
        ),
    }
    (MODELS_DIR / "blade_classifier.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    model_status = "active_development" if metrics["validation_accuracy"] >= 0.55 else "validation_failed"
    model_note = metadata["limitations"]
    if model_status == "validation_failed":
        model_note += " O candidato não atingiu o limiar mínimo interno e não foi ativado no motor de inspeção."

    init_db()
    with connect() as connection:
        if model_status == "active_development":
            connection.execute(
                "UPDATE model_versions SET status = 'baseline' "
                "WHERE model_type = 'convolutional_autoencoder' AND status LIKE 'active%'"
            )
        connection.execute(
            """
            INSERT INTO model_versions (
                version_name, model_type, status, created_at, dataset_snapshot_json, metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_name) DO UPDATE SET status=excluded.status, metrics_json=excluded.metrics_json
            """,
            (
                version_name,
                "supervised_static_blade_classifier",
                model_status,
                now_iso(),
                json.dumps({"dataset": "AeBAD-S/test", "train": len(training), "validation": len(validation)}, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                model_note,
            ),
        )
        connection.commit()
    print(json.dumps({"version": version_name, **metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
