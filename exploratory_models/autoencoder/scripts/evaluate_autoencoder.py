from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATASETS_DIR, MODELS_DIR  # noqa: E402
from backend.app.models.autoencoder import BladeAutoencoder  # noqa: E402
from backend.app.services.image_analysis import MODEL_INPUT_SIZE  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEST_CLASSES = ("good", "ablation", "breakdown", "fracture", "groove")


def _records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for class_name in TEST_CLASSES:
        class_root = root / class_name
        if not class_root.exists():
            continue
        for path in sorted(class_root.rglob("*")):
            if not path.is_file() or path.name.startswith("._") or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(class_root)
            condition = relative.parts[0] if len(relative.parts) > 1 else "unspecified"
            records.append(
                {
                    "path": path,
                    "source_label": class_name,
                    "condition": condition,
                    "anomaly": 0 if class_name == "good" else 1,
                }
            )
    return records


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _score in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for position, (_score, label) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / position
    return float(precision_sum / positives)


def _binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    positive = sum(labels)
    negative = len(labels) - positive
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(prediction == 1 and label == 1 for prediction, label in zip(predictions, labels))
    tn = sum(prediction == 0 and label == 0 for prediction, label in zip(predictions, labels))
    fp = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels))
    fn = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels))
    return {
        "images": len(labels),
        "normal_images": negative,
        "anomaly_images": positive,
        "auroc": None if (value := _roc_auc(labels, scores)) is None else round(value, 4),
        "average_precision": None if (value := _average_precision(labels, scores)) is None else round(value, 4),
        "threshold_raw_top_error": round(float(threshold), 6),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "anomaly_recall": None if positive == 0 else round(tp / positive, 4),
        "normal_specificity": None if negative == 0 else round(tn / negative, 4),
        "false_positive_rate": None if negative == 0 else round(fp / negative, 4),
        "precision_at_calibration_threshold": None if tp + fp == 0 else round(tp / (tp + fp), 4),
        "accuracy_at_calibration_threshold": round((tp + tn) / max(1, len(labels)), 4),
    }


def _load_model(weights_path: Path) -> BladeAutoencoder:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = BladeAutoencoder()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _report_from_scores(
    scored_records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    weights_name: str,
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scored in scored_records:
        grouped["all"].append(scored)
        grouped[f"condition:{scored['condition']}"] .append(scored)
        grouped[f"label:{scored['source_label']}"] .append(scored)
    results: dict[str, Any] = {}
    for group_name, items in sorted(grouped.items()):
        labels = [int(item["anomaly"]) for item in items]
        scores = [float(item["raw_top_error"]) for item in items]
        results[group_name] = _binary_metrics(labels, scores, threshold)
    return {
        "model_version": metadata.get("version_name"),
        "weights": weights_name,
        "dataset": "AeBAD-S/test",
        "protocol": {
            "positive_class": "ablation, breakdown, fracture, or groove",
            "negative_class": "good",
            "score": "99.2 percentile of smoothed reconstruction absolute error",
            "threshold": "training-normal calibration p99; selected before this test, not optimized on test labels",
            "why_not_map": "the autoencoder produces one image-level anomaly score, not class-labelled detections with matched boxes",
        },
        "metrics": results,
        # Kept only to consolidate CPU-friendly evaluation fragments; these
        # entries contain labels and scores, never image pixels.
        "scored_records": scored_records,
    }


def _score_records(records: list[dict[str, Any]], model: BladeAutoencoder, batch_size: int = 32) -> list[dict[str, Any]]:
    """Score batches together so the full public test stays practical on CPU."""
    scored: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        valid_records: list[dict[str, Any]] = []
        tensors: list[torch.Tensor] = []
        for record in records[start : start + batch_size]:
            image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
            tensors.append(torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div(255.0))
            valid_records.append(record)
        if not tensors:
            continue
        source = torch.stack(tensors)
        with torch.inference_mode():
            reconstruction = model(source)
        errors = torch.mean(torch.abs(source - reconstruction), dim=1).cpu().numpy()
        for record, error in zip(valid_records, errors):
            smoothed = cv2.GaussianBlur(error.astype(np.float32), (0, 0), sigmaX=1.1)
            scored.append(
                {
                    "source_label": str(record["source_label"]),
                    "condition": str(record["condition"]),
                    "anomaly": int(record["anomaly"]),
                    "raw_top_error": float(np.percentile(smoothed, 99.2)),
                }
            )
        if min(start + batch_size, len(records)) % 256 == 0 or start + batch_size >= len(records):
            print(f"avaliadas {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mede o autoencoder como classificador binário normal/anomalia no teste público AeBAD-S."
    )
    parser.add_argument("--weights", type=Path, default=MODELS_DIR / "blade_autoencoder.pt")
    parser.add_argument("--metadata", type=Path, default=MODELS_DIR / "blade_autoencoder.json")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "autoencoder_evaluation_aebad_s.json")
    parser.add_argument("--max-images", type=int, default=0, help="0 avalia todas as imagens disponíveis.")
    parser.add_argument("--offset", type=int, default=0, help="Deslocamento para avaliar um fragmento do teste.")
    parser.add_argument("--aggregate-from", type=Path, nargs="+", help="Ficheiros JSON de fragmentos já avaliados para consolidar.")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if not args.weights.exists() or not args.metadata.exists():
        raise RuntimeError("São necessários pesos e metadados do autoencoder local.")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    threshold = float(metadata.get("calibration", {}).get("top_error_p99", 0.0))
    if threshold <= 0:
        raise RuntimeError("Os metadados não contêm uma calibração top_error_p99 válida.")
    if args.aggregate_from:
        scored_records: list[dict[str, Any]] = []
        for path in args.aggregate_from:
            fragment = json.loads(path.read_text(encoding="utf-8"))
            scored_records.extend(fragment.get("scored_records", []))
        if not scored_records:
            raise RuntimeError("Nenhum resultado de fragmento válido foi encontrado para consolidar.")
        report = _report_from_scores(
            scored_records,
            metadata=metadata,
            weights_name=args.weights.name,
            threshold=threshold,
        )
    else:
        root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "test"
        all_records = _records(root)
        start = max(0, args.offset)
        stop = len(all_records) if not args.max_images else start + args.max_images
        records = all_records[start:stop]
        if not records:
            raise RuntimeError(f"Não foram encontradas imagens de teste AeBAD-S em {root} para este fragmento.")
        model = _load_model(args.weights)
        report = _report_from_scores(
            _score_records(records, model),
            metadata=metadata,
            weights_name=args.weights.name,
            threshold=threshold,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all": report["metrics"].get("all")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
