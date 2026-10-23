"""Render 20 audited DLR test panels for the full-data RT-DETR continuation.

The panels never rerun inference.  They use the frozen test predictions and
the confidence threshold selected only on the official DLR validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import run_dlr_benchmark as benchmark


def _header(image: np.ndarray, text: str, colour: tuple[int, int, int]) -> np.ndarray:
    banner = np.full((68, image.shape[1], 3), (18, 28, 43), dtype=np.uint8)
    cv2.putText(banner, text, (18, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.73, colour, 2, cv2.LINE_AA)
    return np.vstack((banner, image))


def _draw_boxes(
    image: np.ndarray,
    boxes: list[dict[str, float]],
    colour: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    result = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(box[key]) for key in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, 4)
        confidence = box.get("confidence")
        text = label if confidence is None else f"{label} {float(confidence):.2f}"
        cv2.putText(result, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2, cv2.LINE_AA)
    return result


def _panel(image: np.ndarray, width: int = 700) -> np.ndarray:
    height, original_width = image.shape[:2]
    return cv2.resize(image, (width, max(1, round(height * width / original_width))), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", default="full_dlr_continuation_20261023")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    evaluation = json.loads((run_dir / f"rtdetr_{args.name}_evaluation.json").read_text(encoding="utf-8"))
    predictions = json.loads((run_dir / f"rtdetr_{args.name}_test_predictions.json").read_text(encoding="utf-8"))["predictions"]
    threshold = float(evaluation["threshold_calibration"]["selected_confidence_threshold"])
    protocol = benchmark._load_protocol_manifest(run_dir / "protocol_manifest.csv")

    selected: list[tuple[benchmark.DlrRecord, bool]] = []
    errors: list[tuple[benchmark.DlrRecord, bool]] = []
    correct: list[tuple[benchmark.DlrRecord, bool]] = []
    for record in protocol.test:
        prediction = predictions[record.sample_id]
        predicted_dent = bool(float(prediction["score"]) >= threshold)
        item = (record, predicted_dent)
        (correct if predicted_dent == record.has_dent else errors).append(item)
    selected.extend(errors[:5])
    selected.extend(correct[:15])
    if len(selected) < 20:
        selected.extend(errors[5 : 5 + (20 - len(selected))])

    destination = run_dir / f"visual_examples_{args.name}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for position, (record, predicted_dent) in enumerate(selected, start=1):
        original = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Imagem inacessível: {record.image_path}")
        truth = "DENT" if record.has_dent else "NORMAL"
        predicted = "DENT" if predicted_dent else "NORMAL"
        prediction = predictions[record.sample_id]
        truth_boxes = [
            {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
            for x1, y1, x2, y2 in benchmark._ground_truth_boxes(record)
        ]
        kept_boxes = [box for box in prediction["boxes"] if float(box["confidence"]) >= threshold]
        left = _header(
            _panel(_draw_boxes(original, truth_boxes, (61, 214, 110), "REAL")),
            # A fonte Hershey do OpenCV não desenha acentos UTF-8 de forma fiável.
            f"Referencia | {truth}",
            (100, 235, 157),
        )
        right = _header(
            _panel(_draw_boxes(original, kept_boxes, (62, 73, 244), "RT-DETR")),
            f"RT-DETR | {predicted} | score {float(prediction['score']):.3f}",
            (115, 130, 255),
        )
        status = "correto" if predicted_dent == record.has_dent else "erro"
        filename = f"{position:02d}_{status}_{record.image_path.stem}.jpg"
        if not cv2.imwrite(str(destination / filename), np.hstack((left, right)), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Não foi possível escrever {filename}")
        manifest.append(
            {
                "file": filename,
                "sample_id": record.sample_id,
                "ground_truth": truth.lower(),
                "prediction": predicted.lower(),
                "score": float(prediction["score"]),
                "selection": status,
            }
        )
    manifest_path = destination / "seleccao_20_exemplos.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"created": len(manifest), "errors_available": len(errors), "destination": destination.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
