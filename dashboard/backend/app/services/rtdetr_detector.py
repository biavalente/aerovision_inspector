"""Runtime local do RT-DETR DLR usado pelo dashboard.

O checkpoint foi avaliado com um limiar congelado a partir da partição de
validação DLR. Esta camada não faz chamadas de rede e não altera o checkpoint.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import PROJECT_ROOT


RUN_NAME = "full_dlr_continuation_20261023"
MODEL_DIR = PROJECT_ROOT / "models" / "rtdetr"
EVALUATION_PATH = MODEL_DIR / "model_metadata.json"


def _checkpoint_path() -> Path:
    """Return an explicitly supplied checkpoint or the documented local slot.

    We deliberately do not trigger a download here.  The dashboard is an
    offline demo and the repository does not redistribute weights whose source
    dataset and runtime licence must be checked by the user.
    """
    environment_value = os.environ.get("AEROVISION_RTDETR_WEIGHTS", "").strip()
    if environment_value:
        return Path(environment_value).expanduser()
    return MODEL_DIR / "best.pt"
INPUT_SIZE = 384
NMS_IOU = 0.70
MAX_DETECTIONS = 100
MAX_DISPLAY_BOXES = 5
# This narrow visual-review band is deliberately separate from the benchmark
# threshold. The benchmark decision remains score >= operating_threshold.
AMBIGUOUS_BAND = 0.015
_PREDICTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class DetectorResult:
    available: bool
    error: str | None
    version_name: str | None
    operating_threshold: float | None
    score: float | None
    decision: str | None
    decision_margin: float | None
    boxes: list[dict[str, Any]]
    evidence_map: np.ndarray | None
    inference_seconds: float | None
    raw_detection_count: int


def _load_evaluation() -> dict[str, Any]:
    if not EVALUATION_PATH.exists():
        return {}
    try:
        return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def _load_runtime() -> tuple[Any | None, dict[str, Any], str | None]:
    """Load once per backend process; no external download is permitted."""
    evaluation = _load_evaluation()
    threshold = evaluation.get("threshold_calibration", {}).get("selected_confidence_threshold")
    metadata: dict[str, Any] = {
        "version_name": str(evaluation.get("candidate") or RUN_NAME),
        "operating_threshold": float(threshold) if threshold is not None else 0.1786108272594388,
        "input_size": INPUT_SIZE,
        "status": str(evaluation.get("status") or "dashboard_demo_review_only"),
        "benchmark": evaluation.get("test", {}).get("metrics", {}),
    }
    checkpoint_path = _checkpoint_path()
    if not checkpoint_path.exists():
        return None, metadata, (
            "Checkpoint RT-DETR local não encontrado. Coloque best.pt em "
            f"{MODEL_DIR} ou defina AEROVISION_RTDETR_WEIGHTS."
        )
    try:
        from ultralytics import RTDETR

        return RTDETR(str(checkpoint_path)), metadata, None
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return None, metadata, f"Não foi possível carregar o RT-DETR local: {error}"


def _dashboard_box(
    xyxy: np.ndarray,
    confidence: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    x1 = max(0.0, min(float(xyxy[0]), float(image_width)))
    y1 = max(0.0, min(float(xyxy[1]), float(image_height)))
    x2 = max(x1, min(float(xyxy[2]), float(image_width)))
    y2 = max(y1, min(float(xyxy[3]), float(image_height)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return {
        "x": round(x1 / image_width * 100, 2),
        "y": round(y1 / image_height * 100, 2),
        "width": round((x2 - x1) / image_width * 100, 2),
        "height": round((y2 - y1) / image_height * 100, 2),
        "label": "dent",
        "confidence": round(float(confidence), 3),
        "source": "model",
        "model_source": "rtdetr_dlr_full",
    }


def _evidence_from_boxes(image_shape: tuple[int, int], boxes: list[dict[str, Any]]) -> np.ndarray:
    """Spatial evidence based on retained detector proposals, not Grad-CAM.

    RT-DETR's localisable evidence is its box geometry and score. A lightly
    blurred confidence map makes that evidence inspectable as a heatmap without
    implying causal or gradient-based explanation.
    """
    height, width = image_shape
    evidence = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x1 = int(round(float(box["x"]) / 100 * width))
        y1 = int(round(float(box["y"]) / 100 * height))
        x2 = int(round((float(box["x"]) + float(box["width"])) / 100 * width))
        y2 = int(round((float(box["y"]) + float(box["height"])) / 100 * height))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        layer = np.zeros_like(evidence)
        strength = max(0.05, float(box["confidence"]))
        cv2.rectangle(layer, (x1, y1), (x2, y2), strength, thickness=-1)
        sigma = max(3.0, min(float(x2 - x1), float(y2 - y1)) / 5)
        evidence = np.maximum(evidence, cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma, sigmaY=sigma))
    return evidence


def analyse_with_rtdetr(image_bgr: np.ndarray) -> DetectorResult:
    """Run the promoted dashboard detector on a BGR image in the local process."""
    model, metadata, load_error = _load_runtime()
    if model is None:
        return DetectorResult(False, load_error, metadata.get("version_name"), metadata.get("operating_threshold"), None, None, None, [], None, None, 0)

    threshold = float(metadata["operating_threshold"])
    try:
        started = time.perf_counter()
        # Ultralytics model instances are shared by synchronous FastAPI calls;
        # serialising prediction avoids concurrent mutation in the runtime.
        with _PREDICTION_LOCK:
            result = model.predict(
                source=image_bgr,
                imgsz=int(metadata["input_size"]),
                conf=0.001,
                iou=NMS_IOU,
                max_det=MAX_DETECTIONS,
                device="cpu",
                verbose=False,
            )[0]
        elapsed = time.perf_counter() - started
    except (OSError, RuntimeError, ValueError) as error:
        return DetectorResult(False, f"Falha na inferência RT-DETR local: {error}", metadata.get("version_name"), threshold, None, None, None, [], None, None, 0)

    height, width = image_bgr.shape[:2]
    raw_boxes: list[dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for coordinates, confidence, class_index in zip(xyxy, confidences, classes):
            if int(class_index) != 0:
                continue
            item = _dashboard_box(coordinates, float(confidence), width, height)
            if item is not None:
                raw_boxes.append(item)
    raw_boxes.sort(key=lambda box: float(box["confidence"]), reverse=True)
    score = float(raw_boxes[0]["confidence"]) if raw_boxes else 0.0
    retained = [box for box in raw_boxes if float(box["confidence"]) >= threshold]

    if abs(score - threshold) <= AMBIGUOUS_BAND:
        decision = "ambiguous"
        margin = 0.0
    elif score >= threshold:
        decision = "likely_anomaly"
        margin = min(1.0, (score - threshold) / 0.12)
    else:
        decision = "likely_normal"
        margin = min(1.0, (threshold - score) / 0.12)

    return DetectorResult(
        True,
        None,
        metadata.get("version_name"),
        threshold,
        score,
        decision,
        margin,
        retained[:MAX_DISPLAY_BOXES],
        _evidence_from_boxes((height, width), retained[:MAX_DISPLAY_BOXES]),
        elapsed,
        len(raw_boxes),
    )
