"""Evaluate a continued RT-DETR candidate without leaking the sealed DLR test.

The operating threshold is selected only on the official DLR validation
partition recorded in the protocol manifest.  The final test partition is
then evaluated once with that frozen threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import run_dlr_benchmark as benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_RUN = PROJECT_ROOT / "dashboard" / "data" / "runs" / "dlr_benchmark_20261022T152053Z"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_BASE_RUN / "rtdetr" / "full_dlr_continuation_20261023" / "weights" / "best.pt",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--name", default="full_dlr_continuation_20261023")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_BASE_RUN)
    return parser.parse_args()


def _with_threshold(
    predictions: dict[str, dict[str, Any]], threshold: float
) -> dict[str, dict[str, Any]]:
    updated = copy.deepcopy(predictions)
    for prediction in updated.values():
        prediction["image_prediction_dent"] = bool(float(prediction["score"]) >= threshold)
    return updated


def main() -> None:
    args = _args()
    base_run = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint inexistente: {checkpoint}")
    protocol = benchmark._load_protocol_manifest(base_run / "protocol_manifest.csv")
    validation = benchmark._run_rtdetr_test(
        protocol.rtdetr_validation,
        checkpoint,
        image_size=args.image_size,
        confidence_threshold=1.1,
    )
    validation_predictions = validation["predictions"]
    normal_scores = [
        float(validation_predictions[record.sample_id]["score"])
        for record in protocol.rtdetr_validation
        if not record.has_dent
    ]
    dent_scores = [
        float(validation_predictions[record.sample_id]["score"])
        for record in protocol.rtdetr_validation
        if record.has_dent
    ]
    threshold, calibration_metrics = benchmark._choose_image_threshold(normal_scores, dent_scores)
    test = benchmark._run_rtdetr_test(
        protocol.test,
        checkpoint,
        image_size=args.image_size,
        confidence_threshold=threshold,
    )
    metrics = benchmark._evaluate_model(protocol.test, test["predictions"], threshold)
    output = {
        "candidate": args.name,
        "base_checkpoint": checkpoint.as_posix(),
        "status": "evaluation_only_not_dashboard_active",
        "protocol": {
            "train_pool": len(protocol.rtdetr_train),
            "validation_images": len(protocol.rtdetr_validation),
            "validation_normals": sum(not record.has_dent for record in protocol.rtdetr_validation),
            "validation_dents": sum(record.has_dent for record in protocol.rtdetr_validation),
            "sealed_test_images": len(protocol.test),
            "sealed_test_normals": sum(not record.has_dent for record in protocol.test),
            "sealed_test_dents": sum(record.has_dent for record in protocol.test),
        },
        "threshold_calibration": {
            "source": "official DLR validation only",
            "selection_rule": "maximizar F1 dent; desempate por especificidade e menor limiar",
            "selected_confidence_threshold": threshold,
            "metrics": calibration_metrics,
            "score_ranges": {
                "normal": [min(normal_scores), max(normal_scores)],
                "dent": [min(dent_scores), max(dent_scores)],
            },
        },
        "test": {"metrics": metrics, "timing": test["timing"]},
    }
    destination = base_run / f"rtdetr_{args.name}_evaluation.json"
    predictions_destination = base_run / f"rtdetr_{args.name}_test_predictions.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_destination.write_text(
        json.dumps({"predictions": test["predictions"]}, indent=2), encoding="utf-8"
    )
    benchmark._save_confusion_matrix(
        metrics["image_level"],
        base_run / "rtdetr" / f"{args.name}_confusion_matrix.png",
        f"RT-DETR - teste selado - limiar {threshold:.4f}",
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
