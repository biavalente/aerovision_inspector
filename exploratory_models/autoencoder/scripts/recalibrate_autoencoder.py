from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATASETS_DIR, MODELS_DIR  # noqa: E402
from backend.app.db import connect, now_iso  # noqa: E402
from evaluate_autoencoder import IMAGE_SUFFIXES, _load_model, _score_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalibra os limiares do autoencoder usando exatamente o mesmo mapa de erro usado na inferência."
    )
    parser.add_argument("--apply", action="store_true", help="Atualiza os metadados ativos depois de gerar a calibração.")
    parser.add_argument("--offset", type=int, default=0, help="Deslocamento para avaliar um fragmento do treino normal.")
    parser.add_argument("--max-images", type=int, default=0, help="0 usa todas as imagens normais disponíveis.")
    parser.add_argument("--aggregate-from", type=Path, nargs="+", help="Fragmentos de calibração já calculados para consolidar.")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "autoencoder_calibration_aebad_s.json")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    metadata_path = MODELS_DIR / "blade_autoencoder.json"
    weights_path = MODELS_DIR / "blade_autoencoder.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise RuntimeError("Pesos e metadados ativos do autoencoder são necessários.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "train" / "good"
    all_records = [
        {"path": path, "source_label": "good", "condition": "train", "anomaly": 0}
        for path in sorted(train_root.rglob("*"))
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(all_records) < 8:
        raise RuntimeError(f"Não existem imagens normais suficientes em {train_root}.")
    if args.aggregate_from:
        scores: list[float] = []
        for path in args.aggregate_from:
            fragment = json.loads(path.read_text(encoding="utf-8"))
            scores.extend(float(value) for value in fragment.get("scores", []))
    else:
        start = max(0, args.offset)
        stop = len(all_records) if not args.max_images else start + args.max_images
        records = all_records[start:stop]
        if not records:
            raise RuntimeError("O fragmento de calibração não contém imagens.")
        model = _load_model(weights_path)
        scores = [float(record["raw_top_error"]) for record in _score_records(records, model)]
    if not scores:
        raise RuntimeError("Não foi possível calcular qualquer erro de reconstrução para a calibração.")
    calibration = {
        "top_error_p50": float(np.percentile(scores, 50)),
        "top_error_p95": float(np.percentile(scores, 95)),
        "top_error_p99": float(np.percentile(scores, 99)),
    }
    report = {
        "model_version": metadata.get("version_name"),
        "calibration_dataset": "AeBAD-S/train/good",
        "calibration_images": len(scores),
        "available_training_images": len(all_records),
        "old_calibration": metadata.get("calibration", {}),
        "new_calibration": calibration,
        "protocol": "99.2º percentil do mapa de erro absoluto depois do mesmo GaussianBlur usado na inferência",
        "applied": bool(args.apply),
        "scores": scores,
    }
    report_path = args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.apply:
        if len(scores) != len(all_records):
            raise RuntimeError("Só é permitido aplicar uma calibração calculada sobre todas as imagens normais de treino.")
        metadata["calibration"] = calibration
        metadata["calibration_protocol"] = report["protocol"]
        metadata["calibration_corrected_at"] = now_iso()
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        with connect() as connection:
            row = connection.execute(
                "SELECT metrics_json FROM model_versions WHERE version_name = ?", (metadata.get("version_name"),)
            ).fetchone()
            if row:
                metrics = json.loads(row["metrics_json"] or "{}")
                metrics.update({"calibration_recomputed": True, **calibration})
                connection.execute(
                    "UPDATE model_versions SET metrics_json = ?, notes = ? WHERE version_name = ?",
                    (
                        json.dumps(metrics, ensure_ascii=False),
                        "Calibração alinhada com o mapa de erro suavizado usado na inferência; o checkpoint não foi alterado.",
                        metadata.get("version_name"),
                    ),
                )
                connection.commit()
    print(json.dumps({key: value for key, value in {**report, "report_path": str(report_path)}.items() if key != "scores"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
