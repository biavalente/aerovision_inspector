"""Continue the best CPU RT-DETR checkpoint on the full DLR train partition.

The sealed 285-image DLR test partition is not referenced by this program.
It continues from the previously trained hybrid RT-DETR checkpoint, uses the
complete official DLR train split for gradients and the complete official DLR
validation split only for model selection.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ultralytics import RTDETR


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_RUN = PROJECT_ROOT / "dashboard" / "data" / "runs" / "dlr_benchmark_20261022T152053Z"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--name", default="full_dlr_continuation_20261023")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_BASE_RUN,
        help="Diretório do benchmark preparado por run_dlr_benchmark.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    base_run = args.run_dir.resolve()
    checkpoint = base_run / "rtdetr" / "finetune_n_final" / "weights" / "best.pt"
    data_yaml = base_run / "rtdetr_dataset" / "dlr_aircraft_dent.yaml"
    output_root = base_run / "rtdetr"
    if not checkpoint.exists() or not data_yaml.exists():
        raise RuntimeError("Checkpoint ou configuração DLR de origem ausente.")
    config = {
        "started_at": datetime.now(UTC).isoformat(),
        "base_checkpoint": checkpoint.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        "sealed_test_included": False,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.10,
        "augmentations": {
            "scale": 0.15,
            "translate": 0.03,
            "fliplr": 0.5,
            "hsv_s": 0.06,
            "hsv_v": 0.08,
            "mosaic": 0.0,
            "mixup": 0.0,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{args.name}_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model = RTDETR(str(checkpoint))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device="cpu",
        workers=0,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.10,
        cos_lr=True,
        warmup_epochs=0.5,
        patience=2,
        box=7.5,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.06,
        hsv_v=0.08,
        erasing=0.0,
        translate=0.03,
        scale=0.15,
        cache=False,
        amp=False,
        deterministic=True,
        seed=20261023,
        save_period=1,
        plots=True,
        project=str(output_root),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    weights = output_root / args.name / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    chosen = best if best.exists() else last
    if not chosen.exists():
        raise RuntimeError(f"O treino não criou pesos em {weights}.")
    print(json.dumps({"checkpoint": chosen.as_posix(), "config": config}, ensure_ascii=False))


if __name__ == "__main__":
    main()
