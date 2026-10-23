"""Generate an auditable result bundle for the DLR autoencoder and RT-DETR.

The script deliberately evaluates the exact, frozen inference outputs from the
sealed DLR test split.  It therefore never retrains a model, changes a
threshold, or touches the test set while creating presentation artefacts.

It writes a single folder containing: image-level confusion matrices, AUROC,
anomaly average precision, mAP@0.50 localisation metrics, CSV/JSON summaries,
and 20 three-panel visual examples (reference, autoencoder and RT-DETR).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "training" / "rtdetr" / "scripts"))

import run_dlr_benchmark as benchmark


BASE_RUN = PROJECT_ROOT / "dashboard" / "data" / "runs" / "dlr_benchmark_20261022T152053Z"
DEFAULT_OUTPUT_DIR = BASE_RUN / "comparative_results_autoencoder_vs_rtdetr_20261023"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=BASE_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta a criar/actualizar com os resultados comparativos.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Artefacto inexistente: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Formato JSON inesperado: {path}")
    return value


def _as_prediction_map(value: Any, *, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Previsões inválidas para {name}.")
    result: dict[str, dict[str, Any]] = {}
    for sample_id, prediction in value.items():
        if not isinstance(sample_id, str) or not isinstance(prediction, dict):
            raise RuntimeError(f"Previsão inválida para {name}: {sample_id!r}")
        if "score" not in prediction or "boxes" not in prediction:
            raise RuntimeError(f"Previsão incompleta para {name}: {sample_id}")
        result[sample_id] = prediction
    return result


def _assert_coverage(
    test_records: Sequence[benchmark.DlrRecord],
    predictions: dict[str, dict[str, Any]],
    *,
    name: str,
) -> None:
    expected = {record.sample_id for record in test_records}
    actual = set(predictions)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        detail: list[str] = []
        if missing:
            detail.append(f"faltam {len(missing)}")
        if unexpected:
            detail.append(f"sobram {len(unexpected)}")
        raise RuntimeError(f"Cobertura do teste inválida para {name}: {', '.join(detail)}.")


def _with_frozen_threshold(
    predictions: dict[str, dict[str, Any]], threshold: float
) -> dict[str, dict[str, Any]]:
    """Create thresholded image decisions without modifying the saved outputs."""

    prepared = copy.deepcopy(predictions)
    for prediction in prepared.values():
        prediction["image_prediction_dent"] = bool(float(prediction["score"]) >= threshold)
    return prepared


def _roc_points(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    positives = int(sum(labels))
    negatives = len(labels) - positives
    if not labels or not positives or not negatives:
        raise RuntimeError("AUROC requer exemplos normais e exemplos com amolgadela.")
    ordered = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    sorted_scores = np.asarray(scores, dtype=np.float64)[ordered]
    sorted_labels = np.asarray(labels, dtype=np.int8)[ordered]
    true_positive = false_positive = 0
    tpr = [0.0]
    fpr = [0.0]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        true_positive += int(np.sum(group == 1))
        false_positive += int(np.sum(group == 0))
        tpr.append(true_positive / positives)
        fpr.append(false_positive / negatives)
        start = end
    return np.asarray(fpr), np.asarray(tpr)


def _trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    return float(sum((x[index] - x[index - 1]) * (y[index] + y[index - 1]) / 2 for index in range(1, len(x))))


def _average_precision_anomaly(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = int(sum(labels))
    if not labels or not positives:
        raise RuntimeError("Average precision requer pelo menos uma amolgadela.")
    ordered = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ordered_labels = np.asarray(labels, dtype=np.int8)[ordered]
    cumulative_true = np.cumsum(ordered_labels == 1)
    cumulative_all = np.arange(1, len(ordered_labels) + 1)
    recall = np.concatenate(([0.0], cumulative_true / positives))
    precision = np.concatenate(([1.0], cumulative_true / cumulative_all))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _ground_truth_by_sample(
    test_records: Sequence[benchmark.DlrRecord],
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Read each DLR mask once; mAP otherwise repeats thousands of disk reads."""

    return {record.sample_id: benchmark._ground_truth_boxes(record) for record in test_records}


def _match_counts_cached(
    test_records: Sequence[benchmark.DlrRecord],
    predictions: dict[str, dict[str, Any]],
    confidence_threshold: float,
    iou_threshold: float,
    ground_truths: dict[str, list[tuple[int, int, int, int]]],
) -> dict[str, int]:
    true_positive = false_positive = false_negative = 0
    for record in test_records:
        targets = ground_truths[record.sample_id]
        predicted = [
            (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]), float(box["confidence"]))
            for box in predictions[record.sample_id]["boxes"]
            if float(box["confidence"]) >= confidence_threshold
        ]
        matched: set[int] = set()
        for box in sorted(predicted, key=lambda item: item[4], reverse=True):
            best_index = -1
            best_iou = 0.0
            for index, target in enumerate(targets):
                if index not in matched and (overlap := benchmark._iou(box[:4], target)) > best_iou:
                    best_iou, best_index = overlap, index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched.add(best_index)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(targets) - len(matched)
    return {"tp": true_positive, "fp": false_positive, "fn": false_negative}


def _average_precision_detection_cached(
    test_records: Sequence[benchmark.DlrRecord],
    predictions: dict[str, dict[str, Any]],
    iou_threshold: float,
    ground_truths: dict[str, list[tuple[int, int, int, int]]],
) -> float:
    total = sum(len(boxes) for boxes in ground_truths.values())
    if not total:
        return 0.0
    flattened: list[tuple[float, str, tuple[float, float, float, float]]] = []
    for record in test_records:
        for box in predictions[record.sample_id]["boxes"]:
            flattened.append(
                (
                    float(box["confidence"]),
                    record.sample_id,
                    (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                )
            )
    flattened.sort(reverse=True, key=lambda item: item[0])
    matched: dict[str, set[int]] = {}
    true_positive: list[int] = []
    false_positive: list[int] = []
    for _confidence, sample_id, box in flattened:
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(ground_truths[sample_id]):
            if index not in matched.setdefault(sample_id, set()) and (overlap := benchmark._iou(box, target)) > best_iou:
                best_iou, best_index = overlap, index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[sample_id].add(best_index)
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)
    if not true_positive:
        return 0.0
    cumulative_true = np.cumsum(true_positive)
    cumulative_false = np.cumsum(false_positive)
    recalls = cumulative_true / total
    precisions = cumulative_true / np.maximum(1, cumulative_true + cumulative_false)
    return float(
        np.mean(
            [
                np.max(precisions[recalls >= recall]) if np.any(recalls >= recall) else 0.0
                for recall in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def _evaluate_model_cached(
    test_records: Sequence[benchmark.DlrRecord],
    predictions: dict[str, dict[str, Any]],
    confidence_threshold: float,
    ground_truths: dict[str, list[tuple[int, int, int, int]]],
) -> dict[str, Any]:
    labels = [int(record.has_dent) for record in test_records]
    decisions = [bool(predictions[record.sample_id]["image_prediction_dent"]) for record in test_records]
    image_metrics = benchmark._image_classification_metrics(labels, decisions)
    counts = _match_counts_cached(test_records, predictions, confidence_threshold, 0.50, ground_truths)
    detection_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    detection_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    ap_by_iou = {
        f"{threshold:.2f}": _average_precision_detection_cached(test_records, predictions, threshold, ground_truths)
        for threshold in benchmark.TEST_IOU_THRESHOLDS
    }
    return {
        "image_level": image_metrics,
        "detection_level_iou_0_50": {
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
            "precision": detection_precision,
            "recall": detection_recall,
        },
        "map_50": ap_by_iou["0.50"],
        "map_50_95": float(np.mean(list(ap_by_iou.values()))),
        "ap_by_iou": ap_by_iou,
    }


def _model_metrics(
    test_records: Sequence[benchmark.DlrRecord],
    predictions: dict[str, dict[str, Any]],
    *,
    name: str,
    role: str,
    image_threshold: float,
    localisation_threshold: float,
    threshold_description: str,
    timing: dict[str, Any],
    ground_truths: dict[str, list[tuple[int, int, int, int]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[np.ndarray, np.ndarray]]:
    thresholded = _with_frozen_threshold(predictions, image_threshold)
    labels = [int(record.has_dent) for record in test_records]
    scores = [float(thresholded[record.sample_id]["score"]) for record in test_records]
    fpr, tpr = _roc_points(labels, scores)
    evaluated = _evaluate_model_cached(test_records, thresholded, localisation_threshold, ground_truths)
    return (
        {
            "model": name,
            "role": role,
            "image_score": "Maior valor indica maior evidência de amolgadela.",
            "image_threshold": image_threshold,
            "localisation_threshold": localisation_threshold,
            "threshold_description": threshold_description,
            "image_level": evaluated["image_level"],
            "auroc": _trapezoid_area(fpr, tpr),
            "anomaly_average_precision": _average_precision_anomaly(labels, scores),
            "localisation": {
                "detection_level_iou_0_50": evaluated["detection_level_iou_0_50"],
                "map_50": evaluated["map_50"],
                "map_50_95": evaluated["map_50_95"],
                "ap_by_iou": evaluated["ap_by_iou"],
            },
            "timing_seconds": timing,
        },
        thresholded,
        (fpr, tpr),
    )


def _save_roc_chart(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], destination: Path) -> None:
    """Save a lightweight ROC plot without adding a plotting dependency."""

    width, height = 1080, 700
    left, right, top, bottom = 118, 58, 92, 104
    canvas = np.full((height, width, 3), (250, 252, 255), dtype=np.uint8)
    cv2.putText(canvas, "ROC por imagem | teste DLR selado", (42, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (26, 43, 63), 2, cv2.LINE_AA)
    x0, y0 = left, height - bottom
    x1, y1 = width - right, top
    cv2.rectangle(canvas, (x0, y1), (x1, y0), (186, 197, 210), 2)
    for fraction in np.linspace(0.0, 1.0, 6):
        x = int(round(x0 + fraction * (x1 - x0)))
        y = int(round(y0 - fraction * (y0 - y1)))
        cv2.line(canvas, (x, y1), (x, y0), (226, 232, 239), 1)
        cv2.line(canvas, (x0, y), (x1, y), (226, 232, 239), 1)
        cv2.putText(canvas, f"{fraction:.1f}", (x - 12, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65, 81, 101), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{fraction:.1f}", (x0 - 46, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65, 81, 101), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0, y0), (x1, y1), (175, 185, 196), 2, cv2.LINE_AA)
    colours = {"Autoencoder": (20, 140, 255), "RT-DETR": (72, 77, 225)}
    legend_y = 85
    for name, (fpr, tpr, auc) in curves.items():
        points = np.array(
            [[int(round(x0 + x * (x1 - x0))), int(round(y0 - y * (y0 - y1)))] for x, y in zip(fpr, tpr)],
            dtype=np.int32,
        )
        colour = colours.get(name, (80, 140, 80))
        cv2.polylines(canvas, [points], False, colour, 3, cv2.LINE_AA)
        text = f"{name}  AUROC={auc:.3f}"
        text_x = 530 if name == "Autoencoder" else 770
        cv2.line(canvas, (text_x - 32, legend_y - 5), (text_x - 8, legend_y - 5), colour, 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (text_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (37, 52, 71), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Taxa de falsos positivos", (420, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (37, 52, 71), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Taxa de verdadeiros positivos", (22, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (37, 52, 71), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Não foi possível criar {destination}")


def _filtered_boxes(prediction: dict[str, Any], threshold: float, limit: int = 5) -> list[dict[str, Any]]:
    boxes = [box for box in prediction["boxes"] if float(box.get("confidence", 0.0)) >= threshold]
    return sorted(boxes, key=lambda box: float(box.get("confidence", 0.0)), reverse=True)[:limit]


def _draw_boxes(
    image: np.ndarray,
    boxes: Sequence[dict[str, Any]],
    colour: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    drawn = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(float(box[key])) for key in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(drawn, (x1, y1), (x2, y2), colour, 4)
        confidence = box.get("confidence")
        text = label if confidence is None else f"{label} {float(confidence):.2f}"
        cv2.putText(drawn, text, (x1, max(26, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.67, colour, 2, cv2.LINE_AA)
    return drawn


def _header(image: np.ndarray, text: str, colour: tuple[int, int, int]) -> np.ndarray:
    banner = np.full((62, image.shape[1], 3), (19, 30, 46), dtype=np.uint8)
    cv2.putText(banner, text, (16, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
    return np.vstack((banner, image))


def _scale(image: np.ndarray, width: int = 560) -> np.ndarray:
    height, source_width = image.shape[:2]
    return cv2.resize(image, (width, max(1, round(height * width / source_width))), interpolation=cv2.INTER_AREA)


def _prediction_label(prediction: dict[str, Any]) -> str:
    return "DENT" if bool(prediction["image_prediction_dent"]) else "NORMAL"


def _example_groups(
    records: Sequence[benchmark.DlrRecord],
    auto_predictions: dict[str, dict[str, Any]],
    detector_predictions: dict[str, dict[str, Any]],
) -> list[benchmark.DlrRecord]:
    """Select five informative mistakes plus 15 correctly classified cases."""

    groups: dict[str, list[benchmark.DlrRecord]] = {
        "both_wrong": [],
        "ae_wrong_detector_correct": [],
        "ae_correct_detector_wrong": [],
        "both_correct_dent": [],
        "both_correct_normal": [],
    }
    for record in records:
        ae_correct = bool(auto_predictions[record.sample_id]["image_prediction_dent"]) == record.has_dent
        detector_correct = bool(detector_predictions[record.sample_id]["image_prediction_dent"]) == record.has_dent
        if not ae_correct and not detector_correct:
            groups["both_wrong"].append(record)
        elif not ae_correct:
            groups["ae_wrong_detector_correct"].append(record)
        elif not detector_correct:
            groups["ae_correct_detector_wrong"].append(record)
        elif record.has_dent:
            groups["both_correct_dent"].append(record)
        else:
            groups["both_correct_normal"].append(record)

    selected: list[benchmark.DlrRecord] = []
    for name in ("both_wrong", "ae_wrong_detector_correct", "ae_correct_detector_wrong"):
        if groups[name]:
            selected.append(groups[name].pop(0))
    remaining_errors = groups["both_wrong"] + groups["ae_wrong_detector_correct"] + groups["ae_correct_detector_wrong"]
    for record in remaining_errors:
        if len(selected) >= 5:
            break
        selected.append(record)
    for index in range(max(len(groups["both_correct_dent"]), len(groups["both_correct_normal"]))):
        for name in ("both_correct_dent", "both_correct_normal"):
            if index < len(groups[name]) and len(selected) < 20:
                selected.append(groups[name][index])
    if len(selected) < 20:
        for record in remaining_errors:
            if record not in selected:
                selected.append(record)
            if len(selected) >= 20:
                break
    return selected[:20]


def _write_examples(
    records: Sequence[benchmark.DlrRecord],
    auto_predictions: dict[str, dict[str, Any]],
    detector_predictions: dict[str, dict[str, Any]],
    *,
    auto_pixel_threshold: float,
    detector_threshold: float,
    destination: Path,
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    selected = _example_groups(records, auto_predictions, detector_predictions)
    manifest: list[dict[str, Any]] = []
    for position, record in enumerate(selected, start=1):
        original = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Imagem inacessível: {record.image_path}")
        truth_boxes = [
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for x1, y1, x2, y2 in benchmark._ground_truth_boxes(record)
        ]
        auto = auto_predictions[record.sample_id]
        detector = detector_predictions[record.sample_id]
        truth = "DENT" if record.has_dent else "NORMAL"
        auto_label = _prediction_label(auto)
        detector_label = _prediction_label(detector)
        reference_panel = _header(
            _scale(_draw_boxes(original, truth_boxes, (58, 210, 102), "REAL")),
            f"Referencia | real: {truth}",
            (98, 235, 155),
        )
        auto_panel = _header(
            _scale(_draw_boxes(original, _filtered_boxes(auto, auto_pixel_threshold), (25, 156, 255), "AE")),
            f"Autoencoder | previsto: {auto_label} | score {float(auto['score']):.3f}",
            (112, 194, 255),
        )
        detector_panel = _header(
            _scale(_draw_boxes(original, _filtered_boxes(detector, detector_threshold), (62, 74, 245), "RT-DETR")),
            f"RT-DETR | previsto: {detector_label} | score {float(detector['score']):.3f}",
            (116, 131, 255),
        )
        ae_correct = bool(auto["image_prediction_dent"]) == record.has_dent
        detector_correct = bool(detector["image_prediction_dent"]) == record.has_dent
        status = "correto" if ae_correct and detector_correct else "erro"
        filename = f"{position:02d}_{status}_{record.image_path.stem}.jpg"
        if not cv2.imwrite(str(destination / filename), np.hstack((reference_panel, auto_panel, detector_panel)), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Não foi possível criar a imagem de exemplo: {filename}")
        manifest.append(
            {
                "file": filename,
                "sample_id": record.sample_id,
                "ground_truth": truth.lower(),
                "autoencoder": {"prediction": auto_label.lower(), "score": float(auto["score"]), "correct": ae_correct},
                "rtdetr": {"prediction": detector_label.lower(), "score": float(detector["score"]), "correct": detector_correct},
                "selection": status,
            }
        )
    return manifest


def _write_per_image_csv(
    records: Sequence[benchmark.DlrRecord],
    auto_predictions: dict[str, dict[str, Any]],
    detector_predictions: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "ground_truth",
                "autoencoder_score",
                "autoencoder_prediction",
                "rtdetr_score",
                "rtdetr_prediction",
            ),
        )
        writer.writeheader()
        for record in records:
            auto = auto_predictions[record.sample_id]
            detector = detector_predictions[record.sample_id]
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "ground_truth": "dent" if record.has_dent else "normal",
                    "autoencoder_score": f"{float(auto['score']):.8f}",
                    "autoencoder_prediction": _prediction_label(auto).lower(),
                    "rtdetr_score": f"{float(detector['score']):.8f}",
                    "rtdetr_prediction": _prediction_label(detector).lower(),
                }
            )


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _write_metrics_csv(metrics: Sequence[dict[str, Any]], destination: Path, sample_count: int) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "modelo",
                "amostras_teste",
                "tn",
                "fp",
                "fn",
                "tp",
                "accuracy",
                "precisao_dent",
                "recall_dent",
                "f1_dent",
                "especificidade_normal",
                "auroc",
                "average_precision_anomalia",
                "mAP_50",
                "mAP_50_95",
                "precisao_localizacao_iou_0_50",
                "recall_localizacao_iou_0_50",
                "inferencia_media_s",
                "inferencia_min_s",
                "inferencia_max_s",
            ),
        )
        writer.writeheader()
        for item in metrics:
            image = item["image_level"]
            localisation = item["localisation"]
            detection = localisation["detection_level_iou_0_50"]
            timing = item["timing_seconds"]
            writer.writerow(
                {
                    "modelo": item["model"],
                    "amostras_teste": sample_count,
                    "tn": int(image["true_negative"]),
                    "fp": int(image["false_positive"]),
                    "fn": int(image["false_negative"]),
                    "tp": int(image["true_positive"]),
                    "accuracy": f"{image['accuracy']:.6f}",
                    "precisao_dent": f"{image['precision_dent']:.6f}",
                    "recall_dent": f"{image['recall_dent']:.6f}",
                    "f1_dent": f"{image['f1_dent']:.6f}",
                    "especificidade_normal": f"{image['specificity_normal']:.6f}",
                    "auroc": f"{item['auroc']:.6f}",
                    "average_precision_anomalia": f"{item['anomaly_average_precision']:.6f}",
                    "mAP_50": f"{localisation['map_50']:.6f}",
                    "mAP_50_95": f"{localisation['map_50_95']:.6f}",
                    "precisao_localizacao_iou_0_50": f"{detection['precision']:.6f}",
                    "recall_localizacao_iou_0_50": f"{detection['recall']:.6f}",
                    "inferencia_media_s": f"{timing['mean_seconds']:.6f}",
                    "inferencia_min_s": f"{timing['min_seconds']:.6f}",
                    "inferencia_max_s": f"{timing['max_seconds']:.6f}",
                }
            )


def _write_report(
    metrics: Sequence[dict[str, Any]],
    *,
    output: Path,
    test_count: int,
    test_normals: int,
    test_dents: int,
) -> None:
    auto, detector = metrics
    rows = []
    for label, auto_value, detector_value in (
        ("Accuracy por imagem", _percent(auto["image_level"]["accuracy"]), _percent(detector["image_level"]["accuracy"])),
        ("Precisão (dent)", _percent(auto["image_level"]["precision_dent"]), _percent(detector["image_level"]["precision_dent"])),
        ("Recall (dent)", _percent(auto["image_level"]["recall_dent"]), _percent(detector["image_level"]["recall_dent"])),
        ("F1 (dent)", _percent(auto["image_level"]["f1_dent"]), _percent(detector["image_level"]["f1_dent"])),
        ("Especificidade normal", _percent(auto["image_level"]["specificity_normal"]), _percent(detector["image_level"]["specificity_normal"])),
        ("AUROC", f"{auto['auroc']:.3f}", f"{detector['auroc']:.3f}"),
        ("AP de anomalia", f"{auto['anomaly_average_precision']:.3f}", f"{detector['anomaly_average_precision']:.3f}"),
        ("mAP@0.50 (IoU 50%)", _percent(auto["localisation"]["map_50"]), _percent(detector["localisation"]["map_50"])),
        ("mAP@0.50:0.95", _percent(auto["localisation"]["map_50_95"]), _percent(detector["localisation"]["map_50_95"])),
        ("Tempo médio / imagem", f"{auto['timing_seconds']['mean_seconds']:.3f} s", f"{detector['timing_seconds']['mean_seconds']:.3f} s"),
        ("Tempo mínimo–máximo", f"{auto['timing_seconds']['min_seconds']:.3f}–{auto['timing_seconds']['max_seconds']:.3f} s", f"{detector['timing_seconds']['min_seconds']:.3f}–{detector['timing_seconds']['max_seconds']:.3f} s"),
    ):
        rows.append(f"| {label} | {auto_value} | {detector_value} |")
    auto_image = auto["image_level"]
    detector_image = detector["image_level"]
    report = f"""# Resultados comparativos — Autoencoder vs. RT-DETR

## Protocolo

- **Teste DLR selado:** {test_count} imagens ({test_normals} normais; {test_dents} com amolgadela).
- **Sem fuga de dados:** as imagens deste teste não foram usadas no treino nem na calibração de limiar.
- **Reprodutibilidade:** esta pasta recalcula as métricas a partir das previsões congeladas do teste; não retreina modelos nem ajusta limiares.
- **Limiar do autoencoder:** {auto['image_threshold']:.6f} no score de reconstrução; as caixas residuais usam {auto['localisation_threshold']:.6f}.
- **Limiar RT-DETR:** {detector['image_threshold']:.6f} na confiança máxima da caixa.

## Métricas por imagem e localização

| Métrica | Autoencoder | RT-DETR |
| --- | ---: | ---: |
{chr(10).join(rows)}

### Matrizes de confusão por imagem

| Modelo | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Autoencoder | {int(auto_image['true_negative'])} | {int(auto_image['false_positive'])} | {int(auto_image['false_negative'])} | {int(auto_image['true_positive'])} |
| RT-DETR | {int(detector_image['true_negative'])} | {int(detector_image['false_positive'])} | {int(detector_image['false_negative'])} | {int(detector_image['true_positive'])} |

## Leitura correcta dos resultados

O **autoencoder** é um modelo de normalidade: o seu score é erro de reconstrução e as suas caixas são componentes de residual. Assim, o seu `mAP@0.50` é uma medição exploratória dessas regiões residuais, **não** uma alegação de deteção supervisionada equivalente a RT-DETR.

O **RT-DETR** é o detetor supervisionado de amolgadelas. Para a apresentação, usa a sua métrica por imagem (F1/precisão/recall) para explicar o encaminhamento da peça e `mAP@0.50` para explicar a qualidade da localização. Nenhum dos dois resultados representa certificação aeronáutica ou desempenho em campo.

## Ficheiros desta pasta

- `summary.json`: resultados completos, caminhos de origem e limiares congelados.
- `metrics_by_model.csv`: tabela pronta para folha de cálculo.
- `per_image_predictions.csv`: todas as {test_count} previsões por imagem.
- `autoencoder_confusion_matrix.png` e `rtdetr_confusion_matrix.png`: matrizes de confusão.
- `roc_comparativa.png`: curvas ROC e AUROC.
- `20_exemplos/`: 20 imagens, cada uma com referência, previsão do autoencoder e previsão RT-DETR.
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    args = _args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] A carregar o protocolo e as previsões congeladas…", flush=True)
    protocol = benchmark._load_protocol_manifest(run_dir / "protocol_manifest.csv")
    test_records = protocol.test
    if not test_records:
        raise RuntimeError("O protocolo não contém teste final.")

    base_results = _read_json(run_dir / "results.json")
    base_predictions = _read_json(run_dir / "test_predictions.json")
    continued_evaluation = _read_json(run_dir / "rtdetr_full_dlr_continuation_20261023_evaluation.json")
    continued_predictions_document = _read_json(run_dir / "rtdetr_full_dlr_continuation_20261023_test_predictions.json")

    auto_raw = _as_prediction_map(base_predictions.get("autoencoder"), name="autoencoder")
    detector_raw = _as_prediction_map(continued_predictions_document.get("predictions"), name="RT-DETR")
    _assert_coverage(test_records, auto_raw, name="autoencoder")
    _assert_coverage(test_records, detector_raw, name="RT-DETR")

    print("[2/6] A carregar as máscaras de referência uma única vez…", flush=True)
    ground_truths = _ground_truth_by_sample(test_records)
    auto_metadata = base_results["autoencoder"]["metadata"]
    auto_calibration = auto_metadata["calibration"]
    print("[3/6] A calcular métricas do autoencoder…", flush=True)
    auto_metrics, auto_predictions, auto_roc = _model_metrics(
        test_records,
        auto_raw,
        name="Autoencoder",
        role="Modelo de normalidade / score de anomalia por reconstrução",
        image_threshold=float(auto_calibration["image_threshold_selected_on_validation"]),
        localisation_threshold=float(auto_calibration["pixel_threshold_normal_p99_7"]),
        threshold_description="Limiar de score escolhido na validação DLR; limiar de pixel residual escolhido em normais de validação.",
        timing=base_results["autoencoder"]["timing"],
        ground_truths=ground_truths,
    )
    print("[4/6] A calcular métricas do RT-DETR…", flush=True)
    detector_metrics, detector_predictions, detector_roc = _model_metrics(
        test_records,
        detector_raw,
        name="RT-DETR",
        role="Detetor supervisionado de amolgadelas com caixas delimitadoras",
        image_threshold=float(continued_evaluation["threshold_calibration"]["selected_confidence_threshold"]),
        localisation_threshold=float(continued_evaluation["threshold_calibration"]["selected_confidence_threshold"]),
        threshold_description="Limiar de confiança escolhido apenas na partição DLR de validação.",
        timing=continued_evaluation["test"]["timing"],
        ground_truths=ground_truths,
    )

    print("[5/6] A criar matrizes, curva ROC e 20 exemplos…", flush=True)
    benchmark._save_confusion_matrix(
        auto_metrics["image_level"],
        output_dir / "autoencoder_confusion_matrix.png",
        "Autoencoder | teste DLR selado",
    )
    benchmark._save_confusion_matrix(
        detector_metrics["image_level"],
        output_dir / "rtdetr_confusion_matrix.png",
        "RT-DETR | teste DLR selado",
    )
    _save_roc_chart(
        {
            "Autoencoder": (*auto_roc, auto_metrics["auroc"]),
            "RT-DETR": (*detector_roc, detector_metrics["auroc"]),
        },
        output_dir / "roc_comparativa.png",
    )
    examples = _write_examples(
        test_records,
        auto_predictions,
        detector_predictions,
        auto_pixel_threshold=auto_metrics["localisation_threshold"],
        detector_threshold=detector_metrics["localisation_threshold"],
        destination=output_dir / "20_exemplos",
    )
    (output_dir / "seleccao_20_exemplos.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_per_image_csv(test_records, auto_predictions, detector_predictions, output_dir / "per_image_predictions.csv")
    metrics = [auto_metrics, detector_metrics]
    print("[6/6] A escrever tabelas e relatório…", flush=True)
    _write_metrics_csv(metrics, output_dir / "metrics_by_model.csv", len(test_records))
    _write_report(
        metrics,
        output=output_dir / "comparison_report.md",
        test_count=len(test_records),
        test_normals=sum(not record.has_dent for record in test_records),
        test_dents=sum(record.has_dent for record in test_records),
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_mode": "metrics recalculated from frozen sealed-test inference outputs; no retraining or threshold tuning",
        "protocol": {
            "manifest": (run_dir / "protocol_manifest.csv").as_posix(),
            "test_images": len(test_records),
            "test_normals": sum(not record.has_dent for record in test_records),
            "test_dents": sum(record.has_dent for record in test_records),
        },
        "source_artifacts": {
            "autoencoder_results": (run_dir / "results.json").as_posix(),
            "autoencoder_predictions": (run_dir / "test_predictions.json").as_posix(),
            "rtdetr_evaluation": (run_dir / "rtdetr_full_dlr_continuation_20261023_evaluation.json").as_posix(),
            "rtdetr_predictions": (run_dir / "rtdetr_full_dlr_continuation_20261023_test_predictions.json").as_posix(),
        },
        "models": metrics,
        "visual_examples": {"created": len(examples), "directory": "20_exemplos", "manifest": "seleccao_20_exemplos.json"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Resultados comparativos\n\nComeça por abrir `comparison_report.md`. Os dados completos estão em `summary.json` e `metrics_by_model.csv`.\n",
        encoding="utf-8",
    )
    print("Concluído.", flush=True)
    print(
        json.dumps(
            {
                "output": output_dir.as_posix(),
                "test_images": len(test_records),
                "examples": len(examples),
                "autoencoder_auroc": auto_metrics["auroc"],
                "rtdetr_auroc": detector_metrics["auroc"],
                "rtdetr_map_50": detector_metrics["localisation"]["map_50"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
