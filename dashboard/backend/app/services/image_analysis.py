from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from ..config import MODELS_DIR, RUNS_DIR
from ..models.autoencoder import BladeAutoencoder
from ..models.defect_type_resnet import DEFECT_TYPE_CLASSES, ROI_CLASS_NAMES, DefectTypeResNet18
from ..defect_labels import portuguese_label
from .rtdetr_detector import analyse_with_rtdetr


MODEL_PATH = MODELS_DIR / "blade_autoencoder.pt"
MODEL_METADATA_PATH = MODELS_DIR / "blade_autoencoder.json"
TYPE_CLASSIFIER_PATH = MODELS_DIR / "defect_type_resnet18.pt"
TYPE_CLASSIFIER_METADATA_PATH = MODELS_DIR / "defect_type_resnet18.json"
MODEL_INPUT_SIZE = 192
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)
TYPE_GRID_SCALES = (0.20, 0.30, 0.45, 0.65)
TYPE_GRID_DETECTION_THRESHOLD = 0.70
# This gate is deliberately stricter than the crop-level promotion score.
# A full-image scan tests many candidate regions, so anything below 98% is
# retained as a reviewable complex case rather than presented as a strong find.
TYPE_GRID_STRONG_THRESHOLD = 0.98
# Keep type labels conservative: two different strong visual explanations that
# are within 15 percentage points are a review case, not a decisive label.
TYPE_GRID_CONFLICT_GAP = 0.15


@dataclass
class AnalysisResult:
    ai_state: str
    anomaly_score: float | None
    operational_confidence: float | None
    quality_state: str
    quality: dict[str, Any]
    model_version: str | None
    model_boxes: list[dict[str, Any]]
    xai_bgr: np.ndarray | None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_active_model() -> tuple[nn.Module | None, dict[str, Any], str | None]:
    """Load only a local checkpoint. The service never requests a remote model."""
    metadata = _load_metadata(MODEL_METADATA_PATH)
    if not MODEL_PATH.exists():
        return None, metadata, None
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = BladeAutoencoder()
        model.load_state_dict(state_dict)
        model.eval()
        return model, metadata, "autoencoder"
    except (OSError, RuntimeError, ValueError):
        return None, metadata, None


def _load_active_type_classifier() -> tuple[DefectTypeResNet18 | None, dict[str, Any]]:
    """Load the companion classifier only after it passes its promotion gate.

    It supplies a non-blocking, multi-scale type/rejection signal alongside
    the autoencoder residual map. It never approves or rejects a part on its
    own; the human review remains the final decision.
    """
    metadata = _load_metadata(TYPE_CLASSIFIER_METADATA_PATH)
    if (
        not TYPE_CLASSIFIER_PATH.exists()
        or metadata.get("promotion_status") != "active_type_support"
        or tuple(metadata.get("class_names", ())) != ROI_CLASS_NAMES
    ):
        return None, metadata
    try:
        checkpoint = torch.load(TYPE_CLASSIFIER_PATH, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        classifier = DefectTypeResNet18(num_classes=len(ROI_CLASS_NAMES))
        classifier.load_state_dict(state_dict)
        classifier.eval()
        return classifier, metadata
    except (OSError, RuntimeError, ValueError):
        return None, metadata


def assess_capture_quality(image_bgr: np.ndarray) -> tuple[str, dict[str, Any], float]:
    """Measure image suitability without rejecting or stopping an inspection."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright_fraction = float(np.mean(gray >= 248))
    dark_fraction = float(np.mean(gray <= 8))
    contrast = float(gray.std())

    focus_component = _clip(blur_variance / 120.0)
    glare_component = 1.0 - _clip(bright_fraction / 0.18)
    exposure_component = 1.0 - _clip((bright_fraction + dark_fraction) / 0.35)
    contrast_component = _clip(contrast / 42.0)
    reliability = _clip(
        0.35 * focus_component
        + 0.25 * glare_component
        + 0.25 * exposure_component
        + 0.15 * contrast_component
    )
    quality_state = "capture_quality_warning" if reliability < 0.48 else "acceptable"

    return quality_state, {
        "reliability": round(reliability, 3),
        "blur_laplacian_variance": round(blur_variance, 2),
        "glare_fraction": round(bright_fraction, 4),
        "dark_fraction": round(dark_fraction, 4),
        "contrast_std": round(contrast, 2),
        "note": (
            "A qualidade de captura reduz a confiança operacional; a análise continua."
            if quality_state == "capture_quality_warning"
            else "Indicadores de captura dentro dos limites operacionais do protótipo."
        ),
    }, reliability


def _error_map(image_bgr: np.ndarray, model: BladeAutoencoder) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.inference_mode():
        reconstruction = model(tensor)
    reconstruction_np = reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy()
    source_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    error = np.mean(np.abs(source_np - reconstruction_np), axis=2).astype(np.float32)
    return cv2.GaussianBlur(error, (0, 0), sigmaX=1.1)


def _crop_model_roi(image_bgr: np.ndarray, box: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image_height, image_width = image_bgr.shape[:2]
    x = float(box["x"]) / 100.0 * image_width
    y = float(box["y"]) / 100.0 * image_height
    width = float(box["width"]) / 100.0 * image_width
    height = float(box["height"]) / 100.0 * image_height
    centre_x = x + width / 2
    centre_y = y + height / 2
    # Mirrors the context surrounding the mask-derived crops used in training.
    side = max(max(width, height) * 2.2, min(image_height, image_width) * 0.08, 192.0)
    x1 = max(0, int(round(centre_x - side / 2)))
    y1 = max(0, int(round(centre_y - side / 2)))
    x2 = min(image_width, int(round(centre_x + side / 2)))
    y2 = min(image_height, int(round(centre_y + side / 2)))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("A ROI prevista pelo detetor ficou vazia.")
    return crop, (x1, y1, x2, y2)


def _classifier_tensor(crop_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    resized = cv2.resize(crop_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)


def _type_distribution(probabilities: torch.Tensor) -> dict[str, float]:
    return {
        class_name: round(float(probabilities[index]), 4)
        for index, class_name in enumerate(ROI_CLASS_NAMES)
    }


def _type_classifier_crop_evidence(
    image_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    coordinates: tuple[int, int, int, int],
    model: DefectTypeResNet18,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    x1, y1, x2, y2 = coordinates
    tensor = _classifier_tensor(crop_bgr, int(metadata.get("input_size", 224)))
    temperature = max(0.1, float(metadata.get("metrics", {}).get("temperature", 1.0)))
    model.zero_grad(set_to_none=True)
    logits, activations = model.forward_features(tensor)
    probabilities = torch.softmax(logits / temperature, dim=1)
    predicted_index = int(probabilities.argmax(dim=1).item())
    target_score = logits[0, predicted_index]
    gradients = torch.autograd.grad(target_score, activations)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0).detach().cpu().numpy()
    cam = cam / max(float(cam.max()), 1e-7)
    full_cam = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    full_cam[y1:y2, x1:x2] = cv2.resize(cam, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    detached_probabilities = probabilities.detach()
    distribution = _type_distribution(detached_probabilities[0])
    raw_label = ROI_CLASS_NAMES[predicted_index]
    confidence = float(detached_probabilities[0, predicted_index])
    threshold = float(metadata.get("inconclusive_confidence_threshold", 0.45))
    if raw_label == "normal_roi":
        label = raw_label
    else:
        label = raw_label if confidence >= threshold else "unknown"
    return {
        "label": label,
        "top_class": raw_label,
        "confidence": round(confidence, 3),
        "distribution": distribution,
        "evidence_map": full_cam,
        "roi": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
    }


def _type_classifier_evidence(
    image_bgr: np.ndarray,
    box: dict[str, Any],
    model: DefectTypeResNet18,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    crop, coordinates = _crop_model_roi(image_bgr, box)
    return _type_classifier_crop_evidence(image_bgr, crop, coordinates, model, metadata)


def _grid_starts(length: int, side: int) -> list[int]:
    side = min(side, length)
    step = max(1, side // 2)
    starts = list(range(0, max(1, length - side + 1), step))
    final_start = max(0, length - side)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


def _type_grid_rois(image_width: int, image_height: int) -> list[tuple[int, int, int, int]]:
    """Cover the image with overlapping classifier crops at several scales.

    The classifier was trained on a defect plus context.  Scanning the local
    image avoids asking the low-resolution autoencoder to localise a small
    flaw on a large metal part, while the normal ROI class rejects ordinary
    material texture.  This is intentionally an offline, review-oriented
    operation rather than a real-time detector.
    """
    rois: list[tuple[int, int, int, int]] = []
    for fraction in TYPE_GRID_SCALES:
        side = min(min(image_width, image_height), max(224, int(round(min(image_width, image_height) * fraction))))
        for y1 in _grid_starts(image_height, side):
            for x1 in _grid_starts(image_width, side):
                rois.append((x1, y1, min(image_width, x1 + side), min(image_height, y1 + side)))
    return rois


def _scan_type_classifier(
    image_bgr: np.ndarray,
    model: DefectTypeResNet18,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    image_height, image_width = image_bgr.shape[:2]
    rois = _type_grid_rois(image_width, image_height)
    input_size = int(metadata.get("input_size", 224))
    temperature = max(0.1, float(metadata.get("metrics", {}).get("temperature", 1.0)))
    tensors = [
        _classifier_tensor(image_bgr[y1:y2, x1:x2], input_size)
        for x1, y1, x2, y2 in rois
    ]
    all_probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), 32):
            logits = model(torch.cat(tensors[start : start + 32], dim=0))
            all_probabilities.extend(torch.softmax(logits / temperature, dim=1).cpu())

    threshold = float(metadata.get("inconclusive_confidence_threshold", 0.45))
    predictions: list[dict[str, Any]] = []
    for (x1, y1, x2, y2), probabilities in zip(rois, all_probabilities):
        class_index = int(probabilities.argmax().item())
        top_class = ROI_CLASS_NAMES[class_index]
        confidence = float(probabilities[class_index])
        label = top_class if top_class == "normal_roi" or confidence >= threshold else "unknown"
        predictions.append(
            {
                "label": label,
                "top_class": top_class,
                "confidence": round(confidence, 3),
                "distribution": _type_distribution(probabilities),
                "normal_confidence": round(float(probabilities[0]), 3),
                "roi": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "source": "multi_scale_type_scan",
            }
        )
    return predictions


def _prediction_box(prediction: dict[str, Any], image_width: int, image_height: int) -> dict[str, Any]:
    roi = prediction["roi"]
    return {
        "x": round(float(roi["x"]) / image_width * 100, 2),
        "y": round(float(roi["y"]) / image_height * 100, 2),
        "width": round(float(roi["width"]) / image_width * 100, 2),
        "height": round(float(roi["height"]) / image_height * 100, 2),
        # Persist the stable model identifier.  The API/UI maps it to a
        # Portuguese label, while old versions remain interpretable.
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "type_label": prediction["label"],
        "type_confidence": prediction["confidence"],
        "source": "model",
        "model_source": prediction["source"],
    }


def _anomaly_score(error_map: np.ndarray, metadata: dict[str, Any]) -> tuple[float, dict[str, float]]:
    raw_top = float(np.percentile(error_map, 99.2))
    raw_mean = float(np.mean(error_map))
    raw_p95 = float(np.percentile(error_map, 95))
    calibration = metadata.get("calibration", {})
    normal_p50 = float(calibration.get("top_error_p50", max(raw_mean * 0.75, 1e-5)))
    normal_p95 = float(calibration.get("top_error_p95", max(raw_p95, normal_p50 + 1e-5)))
    normal_p99 = float(calibration.get("top_error_p99", normal_p95 * 1.12 + 1e-5))

    # A p95 normal scores near 0.50 and a p99 normal scores near 0.80.
    # This is an operational anomaly index, never a defect probability.
    scale = max(normal_p99 - normal_p95, 1e-5)
    score = _clip(0.50 + 0.30 * (raw_top - normal_p95) / scale)
    return score, {
        "reconstruction_error_mean": round(raw_mean, 6),
        "reconstruction_error_p95": round(raw_p95, 6),
        "reconstruction_error_top": round(raw_top, 6),
        "normal_reference_top_p95": round(normal_p95, 6),
        "normal_reference_top_p99": round(normal_p99, 6),
    }


def _extract_boxes(error_map: np.ndarray, score: float, allow_low_score: bool = False) -> list[dict[str, Any]]:
    if score < 0.42 and not allow_low_score:
        return []
    threshold = float(np.percentile(error_map, 99.35))
    binary = (error_map >= threshold).astype(np.uint8) * 255
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = error_map.shape
    min_area = max(18, int(width * height * 0.001))
    boxes: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        if int(area) < min_area:
            continue
        boxes.append(
            {
                "x": round(float(x) / width * 100, 2),
                "y": round(float(y) / height * 100, 2),
                "width": round(float(box_width) / width * 100, 2),
                "height": round(float(box_height) / height * 100, 2),
                "label": "visual_anomaly_region",
                "confidence": round(_clip(0.35 + score * 0.62), 3),
                "source": "model",
            }
        )
    boxes.sort(key=lambda item: item["width"] * item["height"], reverse=True)
    return boxes[:5]


def _normalize_evidence(evidence_map: np.ndarray) -> np.ndarray:
    ceiling = float(np.percentile(evidence_map, 99.4))
    floor = float(np.percentile(evidence_map, 5.0))
    return np.clip((evidence_map - floor) / max(ceiling - floor, 1e-7), 0, 1).astype(np.float32)


def _make_xai_overlay(image_bgr: np.ndarray, evidence_map: np.ndarray, caption: str) -> np.ndarray:
    image_height, image_width = image_bgr.shape[:2]
    upscaled = cv2.resize(evidence_map, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    normalized = _normalize_evidence(upscaled)
    if float(normalized.max()) <= 1e-6:
        overlay = image_bgr.copy()
    else:
        heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(image_bgr, 0.65, heatmap, 0.35, 0)
    cv2.putText(
        overlay,
        caption,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        caption,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (24, 24, 24),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _analyse_image_legacy_resnet(image_path: str | Path) -> AnalysisResult:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Não foi possível abrir a imagem para análise.")

    quality_state, quality, reliability = assess_capture_quality(image_bgr)
    model, metadata, _model_kind = _load_active_model()
    if model is None:
        quality["xai_method"] = "Indisponível até existir um modelo local treinado."
        return AnalysisResult(
            ai_state="model_not_ready",
            anomaly_score=None,
            operational_confidence=None,
            quality_state=quality_state,
            quality=quality,
            model_version=None,
            model_boxes=[],
            xai_bgr=None,
        )

    assert isinstance(model, BladeAutoencoder)
    error_map = _error_map(image_bgr, model)
    reconstruction_score, error_metrics = _anomaly_score(error_map, metadata)
    if reconstruction_score >= 0.82:
        base_ai_state = "likely_anomaly"
        base_margin = (reconstruction_score - 0.82) / 0.18
    elif reconstruction_score <= 0.40:
        base_ai_state = "likely_normal"
        base_margin = (0.40 - reconstruction_score) / 0.40
    else:
        base_ai_state = "ambiguous"
        base_margin = 0.0

    type_model, type_metadata = _load_active_type_classifier()
    ai_state = base_ai_state
    margin = base_margin
    anomaly_score = reconstruction_score
    boxes = _extract_boxes(error_map, reconstruction_score)
    evidence_map: np.ndarray = error_map
    caption = "Mapa de evidência: erro de reconstrução local"
    decision_basis = [
        "desvio visual face ao padrão normal aprendido",
        "regiões residuais propostas pelo mapa de reconstrução",
        "indicadores de foco, brilho especular e exposição",
    ]
    quality.update(error_metrics)
    quality["reconstruction_anomaly_index"] = round(reconstruction_score, 3)

    if type_model is not None:
        type_predictions = _scan_type_classifier(image_bgr, type_model, type_metadata)
        type_defects = sorted(
            (
                prediction
                for prediction in type_predictions
                if prediction["top_class"] in DEFECT_TYPE_CLASSES
                and float(prediction["confidence"]) >= TYPE_GRID_DETECTION_THRESHOLD
            ),
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        strongest_by_type: dict[str, dict[str, Any]] = {}
        for prediction in type_defects:
            strongest_by_type.setdefault(str(prediction["top_class"]), prediction)
        ranked_types = sorted(
            strongest_by_type.values(),
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        primary_prediction = ranked_types[0] if ranked_types else None
        contender = ranked_types[1] if len(ranked_types) > 1 else None
        conflict = bool(
            primary_prediction is not None
            and contender is not None
            and float(contender["confidence"]) >= float(primary_prediction["confidence"]) - TYPE_GRID_CONFLICT_GAP
        )
        normal_tile_fraction = (
            sum(prediction["top_class"] == "normal_roi" for prediction in type_predictions)
            / max(1, len(type_predictions))
        )
        visible_predictions = [primary_prediction] if primary_prediction is not None else []
        if conflict and contender is not None:
            visible_predictions.append(contender)
        visible_boxes = [
            _prediction_box(prediction, image_bgr.shape[1], image_bgr.shape[0])
            for prediction in visible_predictions
        ]

        if primary_prediction is not None:
            anomaly_score = float(primary_prediction["confidence"])
            if float(primary_prediction["confidence"]) >= TYPE_GRID_STRONG_THRESHOLD and not conflict:
                type_ai_state = "likely_anomaly"
                type_margin = (anomaly_score - TYPE_GRID_STRONG_THRESHOLD) / (1.0 - TYPE_GRID_STRONG_THRESHOLD)
            else:
                type_ai_state = "ambiguous"
                type_margin = 0.0
        elif normal_tile_fraction >= 0.85:
            anomaly_score = _clip(1.0 - normal_tile_fraction)
            type_ai_state = "likely_normal"
            type_margin = (normal_tile_fraction - 0.85) / 0.15
        else:
            anomaly_score = _clip(1.0 - normal_tile_fraction)
            type_ai_state = "ambiguous"
            type_margin = 0.0

        if type_ai_state == "likely_anomaly":
            ai_state = "likely_anomaly"
            margin = type_margin
            boxes = visible_boxes
        elif type_ai_state == "likely_normal" and base_ai_state != "likely_anomaly":
            ai_state = "likely_normal"
            margin = type_margin
            boxes = []
        else:
            ai_state = "ambiguous"
            margin = 0.0
            boxes = visible_boxes or boxes

        error_full = cv2.resize(error_map, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        if primary_prediction is not None:
            roi = primary_prediction["roi"]
            x1, y1 = int(roi["x"]), int(roi["y"])
            x2 = x1 + int(roi["width"])
            y2 = y1 + int(roi["height"])
            evidence = _type_classifier_crop_evidence(
                image_bgr,
                image_bgr[y1:y2, x1:x2],
                (x1, y1, x2, y2),
                type_model,
                type_metadata,
            )["evidence_map"]
            evidence_map = np.maximum(0.42 * _normalize_evidence(error_full), _normalize_evidence(evidence))
            primary_label = str(primary_prediction["top_class"])
            caption = f"Varredura multi-escala + Grad-CAM: {portuguese_label(primary_label)}"
            quality.update(
                {
                    "predicted_defect_type": primary_label,
                    "predicted_defect_type_confidence": primary_prediction["confidence"],
                    "defect_type_distribution": primary_prediction["distribution"],
                }
            )
            if conflict and contender is not None:
                decision_basis.append(
                    "a varredura encontrou dois tipos visuais fortes e discordantes; é necessária revisão humana"
                )
            else:
                decision_basis.append(
                    f"tipo visual sugerido na região examinada: {portuguese_label(primary_label)} "
                    f"({float(primary_prediction['confidence']):.0%} de confiança calibrada)"
                )
        elif type_ai_state == "likely_normal":
            caption = "Mapa de evidência: varredura rejeitada como normal"
            decision_basis.append(
                f"{normal_tile_fraction:.0%} das regiões examinadas foram rejeitadas como contexto normal"
            )
        else:
            caption = "Mapa de evidência: classificação de tipo inconclusiva"
            decision_basis.append("a varredura de regiões não suportou uma classificação de tipo segura")

        reported_predictions = type_defects[:8]
        if not reported_predictions and type_predictions:
            reported_predictions = sorted(
                type_predictions,
                key=lambda item: float(item["normal_confidence"]),
                reverse=True,
            )[:3]
        quality.update(
            {
                "defect_type_status": "available",
                "defect_type_model_version": type_metadata.get("version_name"),
                "defect_type_predictions": reported_predictions,
                "defect_type_scan": {
                    "candidate_count": len(type_predictions),
                    "normal_tile_fraction": round(normal_tile_fraction, 3),
                    "strongest_by_type": {
                        label: prediction["confidence"]
                        for label, prediction in strongest_by_type.items()
                    },
                    "conflict": conflict,
                },
                "xai_method": (
                    "Varredura multi-escala do classificador local de tipo/rejeição e Grad-CAM na região "
                    "mais relevante, combinado com o mapa residual do autoencoder."
                ),
                "xai_interpretation": (
                    "A varredura local procura padrões de defeito em vários tamanhos; a classe normal rejeita "
                    "textura corrente. O mapa residual acrescenta uma segunda fonte de evidência. "
                    "O resultado mantém-se como apoio à revisão humana."
                ),
                "decision_basis": decision_basis,
            }
        )
    else:
        quality.update(
            {
                "defect_type_status": str(type_metadata.get("promotion_status")) if type_metadata else "not_available",
                "xai_method": "Mapa de erro de reconstrução do autoencoder local.",
                "xai_interpretation": (
                    "As zonas mais quentes são regiões que o modelo normal não conseguiu reconstruir bem; "
                    "são evidência para revisão humana, não uma prova de defeito."
                ),
                "decision_basis": decision_basis,
            }
        )

    operational_confidence = _clip(0.42 + 0.52 * _clip(margin) * reliability, 0.35, 0.96)

    return AnalysisResult(
        ai_state=ai_state,
        anomaly_score=round(anomaly_score, 3),
        operational_confidence=round(operational_confidence, 3),
        quality_state=quality_state,
        quality=quality,
        model_version=str(metadata.get("version_name") or "blade_autoencoder_local"),
        model_boxes=boxes,
        xai_bgr=_make_xai_overlay(image_bgr, evidence_map, caption),
    )


def analyse_image(image_path: str | Path) -> AnalysisResult:
    """Analyse an inspection with the active RT-DETR detector.

    The former ResNet ROI scan remains in this module only so already stored
    historical results remain interpretable. It is not called by the active
    dashboard path. RT-DETR supplies the operational decision, model boxes,
    and the only XAI evidence rendered by the active dashboard.
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Não foi possível abrir a imagem para análise.")

    quality_state, quality, reliability = assess_capture_quality(image_bgr)
    detector = analyse_with_rtdetr(image_bgr)
    if not detector.available:
        quality.update(
            {
                "active_model_family": "rtdetr_dlr_detector",
                "detector_status": "unavailable",
                "detector_model_version": detector.version_name,
                "xai_method": "Indisponível até o checkpoint RT-DETR local poder ser carregado.",
                "xai_interpretation": detector.error or "O RT-DETR local não está disponível.",
                "decision_basis": ["o detetor local obrigatório não pôde ser carregado"],
            }
        )
        return AnalysisResult(
            ai_state="model_not_ready",
            anomaly_score=None,
            operational_confidence=None,
            quality_state=quality_state,
            quality=quality,
            model_version=detector.version_name,
            model_boxes=[],
            xai_bgr=None,
        )

    detector_map = detector.evidence_map
    if detector_map is None:
        detector_map = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    evidence_map = _normalize_evidence(detector_map)
    xai_method = "Mapa espacial do RT-DETR baseado nas caixas e respetivos scores."

    assert detector.score is not None
    assert detector.operating_threshold is not None
    assert detector.decision is not None
    assert detector.decision_margin is not None
    score = float(detector.score)
    threshold = float(detector.operating_threshold)
    if detector.decision == "likely_anomaly":
        detector_basis = (
            f"o score máximo RT-DETR foi {score:.3f}, acima do limiar operacional DLR de {threshold:.3f}"
        )
    elif detector.decision == "likely_normal":
        detector_basis = (
            f"o score máximo RT-DETR foi {score:.3f}, abaixo do limiar operacional DLR de {threshold:.3f}"
        )
    else:
        detector_basis = (
            f"o score máximo RT-DETR foi {score:.3f}, na margem de revisão do limiar operacional DLR de {threshold:.3f}"
        )

    metric_summary = {
        "image_level_f1": 0.8619246862,
        "image_level_precision": 0.8442622951,
        "image_level_recall": 0.8803418803,
        "map_50": 0.2802511988,
        "map_50_95": 0.0846221595,
        "evaluation_scope": "teste DLR selado; não é validação de campo",
    }
    quality.update(
        {
            "active_model_family": "rtdetr_dlr_detector",
            "detector_status": "available",
            "detector_model_version": detector.version_name,
            "detector_label": "dent",
            "detector_confidence": round(score, 3),
            "detector_threshold": round(threshold, 6),
            "detector_box_count": len(detector.boxes),
            "detector_raw_proposal_count": detector.raw_detection_count,
            "detector_inference_seconds": round(float(detector.inference_seconds or 0.0), 3),
            "detector_benchmark": metric_summary,
            "xai_method": xai_method,
            "xai_interpretation": (
                "As zonas quentes representam a evidência espacial das caixas retidas pelo RT-DETR; "
                "não são Grad-CAM nem uma prova causal. A caixa localiza uma hipótese de amolgadela e deve ser confirmada ou corrigida pelo revisor."
            ),
            "decision_basis": [
                detector_basis,
                f"{len(detector.boxes)} caixa(s) acima do limiar foram preservadas para revisão humana",
                "indicadores de foco, brilho especular e exposição ajustam a confiança operacional, não a decisão do detetor",
            ],
        }
    )

    operational_confidence = _clip(0.42 + 0.52 * float(detector.decision_margin) * reliability, 0.35, 0.96)
    return AnalysisResult(
        ai_state=detector.decision,
        anomaly_score=round(score, 3),
        operational_confidence=round(operational_confidence, 3),
        quality_state=quality_state,
        quality=quality,
        model_version=str(detector.version_name or "rtdetr_dlr_full"),
        model_boxes=detector.boxes,
        xai_bgr=_make_xai_overlay(image_bgr, evidence_map, "Evidencia espacial RT-DETR"),
    )


def save_xai_overlay(inspection_id: int, xai_bgr: np.ndarray | None) -> str | None:
    if xai_bgr is None:
        return None
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RUNS_DIR / f"inspection_{inspection_id}_xai.jpg"
    if not cv2.imwrite(str(destination), xai_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError("Não foi possível guardar o mapa de evidência.")
    return str(destination)
