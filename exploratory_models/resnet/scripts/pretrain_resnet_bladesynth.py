from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATA_DIR, MODELS_DIR, WORKSPACE_ROOT  # noqa: E402
from backend.app.db import connect, init_db, now_iso  # noqa: E402
from backend.app.models.defect_type_resnet import DefectTypeResNet18  # noqa: E402
from train_defect_type_classifier import (  # noqa: E402
    CachedRoiDataset,
    RESNET_WEIGHTS_PATH,
    RoiRecord,
    _bbox_from_mask,
    _prepare_cache,
)


MANIFEST_PATH = DATA_DIR / "manifests" / "resnet_spatial_pretrain_bladesynth.csv"
CACHE_VERSION = "bladesynth_spatial_pretrain_v1"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _records(manifest: Path, role: str, max_per_class: int, *, with_bboxes: bool = True) -> list[RoiRecord]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("role") == role]
    if max_per_class:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row.get("original_label", "unknown"), []).append(row)
        rows = [row for label in sorted(grouped) for row in grouped[label][:max_per_class]]
    records: list[RoiRecord] = []
    for row in rows:
        image_path = _resolve(row["image_path"])
        if not image_path.exists():
            continue
        is_normal = row.get("is_normal") == "1"
        mask_path = _resolve(row["mask_path"]) if row.get("mask_path") else None
        bbox = None
        if not is_normal:
            if mask_path is None or not mask_path.exists():
                continue
            bbox = _bbox_from_mask(mask_path) if with_bboxes else None
            if with_bboxes and bbox is None:
                continue
        source_label = row.get("original_label", "unknown").lower()
        records.append(
            RoiRecord(
                image_path=image_path,
                mask_path=mask_path,
                label_index=0 if is_normal else 1,
                condition=f"bladesynth_{source_label}",
                bbox=bbox,
                cache_path=(
                    DATA_DIR
                    / "derived"
                    / CACHE_VERSION
                    / role
                    / ("normal" if is_normal else "defect")
                    / f"{image_path.parent.name}__{image_path.stem}.jpg"
                ),
                normal_variant=0,
            )
        )
    return records


def _evaluate(model: DefectTypeResNet18, loader: DataLoader, loss_fn: nn.Module) -> tuple[dict[str, float], list[list[int]]]:
    confusion = [[0, 0], [0, 0]]
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            losses.append(float(loss_fn(logits, targets).item()))
            for target, prediction in zip(targets.tolist(), logits.argmax(dim=1).tolist()):
                confusion[int(target)][int(prediction)] += 1
    recalls: list[float] = []
    metrics: dict[str, float] = {"validation_loss": round(float(np.mean(losses)), 4)}
    for index, label in enumerate(("normal_synthetic", "defect_synthetic")):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(2) if row != index)
        false_negative = sum(confusion[index][column] for column in range(2) if column != index)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        metrics[f"precision_{label}"] = round(float(precision), 4)
        metrics[f"recall_{label}"] = round(float(recall), 4)
        recalls.append(recall)
    metrics["validation_accuracy"] = round(float((confusion[0][0] + confusion[1][1]) / max(1, sum(sum(row) for row in confusion))), 4)
    metrics["balanced_recall"] = round(float(np.mean(recalls)), 4)
    return metrics, confusion


def _register(version_name: str, metrics: dict[str, Any], snapshot: dict[str, Any], note: str) -> None:
    init_db()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                version_name, model_type, status, created_at, dataset_snapshot_json, metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_name) DO UPDATE SET status=excluded.status, metrics_json=excluded.metrics_json, notes=excluded.notes
            """,
            (
                version_name,
                "resnet18_spatial_pretrain",
                "validation_required",
                now_iso(),
                json.dumps(snapshot, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                note,
            ),
        )
        connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pré-treina uma espinha dorsal ResNet-18 com ROIs normais/defeituosas sintéticas BladeSynth."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Limite equilibrado por classe sintética para ensaio rápido; 0 usa todo o manifesto.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida o manifesto e apresenta as contagens sem criar cache, checkpoint ou versão de modelo.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise RuntimeError("Manifesto BladeSynth ausente. Execute primeiro scripts/build_training_manifests.py.")
    if not RESNET_WEIGHTS_PATH.exists():
        raise RuntimeError("Pesos ImageNet locais ResNet-18 ausentes.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    train_records = _records(args.manifest, "pretrain_train", args.max_per_class, with_bboxes=not args.dry_run)
    validation_records = _records(args.manifest, "pretrain_validation", args.max_per_class, with_bboxes=not args.dry_run)
    if not train_records or not validation_records:
        raise RuntimeError("O manifesto não produziu dados BladeSynth de treino e validação suficientes.")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "candidate_status": "not_created_dry_run",
                    "training_roi_samples": len(train_records),
                    "validation_roi_samples": len(validation_records),
                    "train_by_source_class": dict(Counter(record.condition for record in train_records)),
                    "validation_by_source_class": dict(Counter(record.condition for record in validation_records)),
                    "binary_train_counts": dict(Counter(record.label_index for record in train_records)),
                    "binary_validation_counts": dict(Counter(record.label_index for record in validation_records)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    _prepare_cache(train_records, rebuild=False)
    _prepare_cache(validation_records, rebuild=False)

    counts = Counter(record.label_index for record in train_records)
    weights = torch.DoubleTensor([1.0 / counts[record.label_index] for record in train_records])
    train_loader = DataLoader(
        CachedRoiDataset(train_records, augment=True),
        batch_size=max(1, args.batch_size),
        sampler=WeightedRandomSampler(weights, num_samples=len(train_records), replacement=True),
        num_workers=0,
    )
    validation_loader = DataLoader(
        CachedRoiDataset(validation_records, augment=False),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
    )
    model = DefectTypeResNet18(num_classes=2)
    model.load_imagenet_backbone(RESNET_WEIGHTS_PATH)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_confusion: list[list[int]] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets in train_loader:
            loss = loss_fn(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        metrics, confusion = _evaluate(model, validation_loader, loss_fn)
        if best_metrics is None or metrics["balanced_recall"] > best_metrics["balanced_recall"]:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_confusion = confusion
        print(
            f"epoch {epoch}/{args.epochs}: train_loss={np.mean(losses):.4f} "
            f"validation_balanced_recall={metrics['balanced_recall']:.3f}",
            flush=True,
        )
    if best_state is None or best_metrics is None or best_confusion is None:
        raise RuntimeError("O pré-treino não produziu um estado de modelo válido.")
    model.load_state_dict(best_state)

    version_name = f"bladesynth_resnet18_spatial_pretrain_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot = {
        "dataset": "BladeSynth selected spatial-pretrain manifest",
        "training_roi_samples": len(train_records),
        "validation_roi_samples": len(validation_records),
        "class_counts": {"normal_synthetic": counts.get(0, 0), "defect_synthetic": counts.get(1, 0)},
        "task": "binary synthetic normal/defect spatial pretraining only",
    }
    metadata = {
        "version_name": version_name,
        "model_type": "resnet18_spatial_pretrain",
        "trained_at": now_iso(),
        "source_dataset": "BladeSynth",
        "task": snapshot["task"],
        "metrics": best_metrics,
        "confusion_matrix": best_confusion,
        "promotion": {
            "automatic": False,
            "next_step": "Fine-tune the backbone on the separate AeBAD-S ROI taxonomy, then validate before promotion.",
        },
        "limitations": "Synthetic pretraining is not aircraft operational validation and is never used directly by dashboard inference.",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version_path = MODELS_DIR / f"{version_name}.pt"
    metadata_path = MODELS_DIR / f"{version_name}.json"
    torch.save({"state_dict": model.state_dict(), "metrics": best_metrics}, version_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # This is a reusable backbone candidate, not the active five-class model.
    shutil.copy2(version_path, MODELS_DIR / "bladesynth_resnet18_spatial_pretrain.pt")
    _register(
        version_name,
        best_metrics,
        snapshot,
        "Pré-treino espacial binário em BladeSynth; exige fine-tuning AeBAD-S e validação antes de qualquer uso.",
    )
    print(
        json.dumps(
            {
                "version": version_name,
                "candidate_weights": str(version_path),
                "backbone_checkpoint": str(MODELS_DIR / "bladesynth_resnet18_spatial_pretrain.pt"),
                **best_metrics,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
