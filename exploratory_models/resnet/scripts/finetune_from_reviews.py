from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter, defaultdict
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

from backend.app.config import DATA_DIR, MODELS_DIR  # noqa: E402
from backend.app.db import append_audit_event, connect, init_db  # noqa: E402
from backend.app.defect_labels import TRAINABLE_DEFECT_LABELS, canonical_defect_label  # noqa: E402
from backend.app.models.defect_type_resnet import ROI_CLASS_NAMES, DefectTypeResNet18  # noqa: E402
from train_defect_type_classifier import (  # noqa: E402
    DEFAULT_TRAIN_CONDITIONS,
    DEFAULT_VALIDATION_CONDITIONS,
    CachedRoiDataset,
    RoiRecord,
    _evaluate,
    _fit_temperature,
    _prepare_cache,
    _records_for_conditions,
)


FINE_TUNE_EPOCHS = 5
HUMAN_VALIDATION_FRACTION = 0.20
REPLAY_PER_CLASS = 60
REGRESSION_MACRO_F1_TOLERANCE = 0.02
REGRESSION_RECALL_TOLERANCE = 0.05


def _mark_version(
    connection,
    version_name: str,
    status: str,
    metrics: dict[str, Any],
    notes: str,
    snapshot_updates: dict[str, Any] | None = None,
) -> None:
    row = connection.execute(
        "SELECT dataset_snapshot_json FROM model_versions WHERE version_name = ?", (version_name,)
    ).fetchone()
    snapshot = json.loads(row["dataset_snapshot_json"] or "{}") if row else {}
    if snapshot_updates:
        snapshot.update(snapshot_updates)
    connection.execute(
        """
        UPDATE model_versions
        SET status = ?, dataset_snapshot_json = ?, metrics_json = ?, notes = ?
        WHERE version_name = ?
        """,
        (status, json.dumps(snapshot, ensure_ascii=False), json.dumps(metrics, ensure_ascii=False), notes, version_name),
    )
    connection.commit()


def _latest_human_boxes(connection, inspection_id: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT b.x, b.y, b.width, b.height, b.label
        FROM bounding_boxes AS b
        JOIN annotation_revisions AS r ON r.id = b.revision_id
        WHERE r.inspection_id = ?
          AND r.revision_number = (
              SELECT MAX(revision_number)
              FROM annotation_revisions
              WHERE inspection_id = ?
          )
        ORDER BY b.id
        """,
        (inspection_id, inspection_id),
    ).fetchall()
    return [dict(row) for row in rows]


def _percent_box_to_pixels(box: dict[str, Any], image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    x = int(round(float(box["x"]) / 100 * image_width))
    y = int(round(float(box["y"]) / 100 * image_height))
    width = int(round(float(box["width"]) / 100 * image_width))
    height = int(round(float(box["height"]) / 100 * image_height))
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(1, min(width, image_width - x))
    height = max(1, min(height, image_height - y))
    return (x, y, width, height) if width > 0 and height > 0 else None


def _human_records(connection, ids: list[int], version_name: str) -> tuple[dict[int, list[RoiRecord]], list[int]]:
    """Turn rejected/complex human decisions into typed ROI records.

    The normal class is deliberately absent: approved images now belong only
    to the primary autoencoder loop. A rejected or complex case teaches this
    classifier only when the reviewer has explicitly marked a supported defect
    type. Untyped complex cases remain in the auditable 250-case cohort but
    never acquire an invented supervision label.
    """
    if not ids:
        return {}, []
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        f"SELECT id, source_path, final_decision FROM inspections WHERE id IN ({placeholders})", ids
    ).fetchall()
    by_case: dict[int, list[RoiRecord]] = defaultdict(list)
    usable_ids: list[int] = []
    for row in rows:
        inspection_id = int(row["id"])
        image_path = Path(str(row["source_path"]))
        if not image_path.exists():
            continue
        decision = str(row["final_decision"])
        if decision in {"rejected_visual", "escalated_ndt"}:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            for index, box in enumerate(_latest_human_boxes(connection, inspection_id), start=1):
                label = canonical_defect_label(str(box["label"]))
                if label not in TRAINABLE_DEFECT_LABELS:
                    continue
                bbox = _percent_box_to_pixels(box, width, height)
                if bbox is None:
                    continue
                by_case[inspection_id].append(
                    RoiRecord(
                        image_path=image_path,
                        mask_path=None,
                        label_index=ROI_CLASS_NAMES.index(label),
                        condition="human_review",
                        bbox=bbox,
                        cache_path=DATA_DIR / "derived" / "active_learning" / version_name / label / f"case_{inspection_id}_box_{index}.jpg",
                    )
                )
        if by_case[inspection_id]:
            usable_ids.append(inspection_id)
    return by_case, sorted(set(usable_ids))


def _case_split(records_by_case: dict[int, list[RoiRecord]], seed: int = 42) -> tuple[list[RoiRecord], list[RoiRecord], list[int], list[int]]:
    case_ids = sorted(records_by_case)
    random.Random(seed).shuffle(case_ids)
    validation_count = max(1, round(len(case_ids) * HUMAN_VALIDATION_FRACTION)) if len(case_ids) > 1 else 0
    validation_ids = case_ids[:validation_count]
    training_ids = case_ids[validation_count:]
    if not training_ids and validation_ids:
        training_ids, validation_ids = validation_ids, []
    training = [record for case_id in training_ids for record in records_by_case[case_id]]
    validation = [record for case_id in validation_ids for record in records_by_case[case_id]]
    return training, validation, training_ids, validation_ids


def _replay_records() -> list[RoiRecord]:
    all_records = _records_for_conditions("train", DEFAULT_TRAIN_CONDITIONS, use_existing_cache=True)
    grouped: dict[int, list[RoiRecord]] = defaultdict(list)
    for record in all_records:
        grouped[record.label_index].append(record)
    rng = random.Random(42)
    selected: list[RoiRecord] = []
    for label_index in sorted(grouped):
        examples = grouped[label_index]
        rng.shuffle(examples)
        selected.extend(examples[:REPLAY_PER_CLASS])
    return selected


def _observed_macro_f1(metrics: dict[str, float], records: list[RoiRecord]) -> float:
    labels = sorted({record.label_index for record in records})
    if not labels:
        return 0.0
    return float(np.mean([metrics[f"f1_{ROI_CLASS_NAMES[index]}"] for index in labels]))


def _load_active_metadata() -> dict[str, Any]:
    path = MODELS_DIR / "defect_type_resnet18.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _flat_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": float(value) for key, value in metrics.items()}


def _run(version_name: str) -> None:
    init_db()
    with connect() as connection:
        version = connection.execute(
            "SELECT dataset_snapshot_json FROM model_versions WHERE version_name = ?", (version_name,)
        ).fetchone()
        if not version:
            raise RuntimeError("Versão candidata não encontrada.")
        snapshot = json.loads(version["dataset_snapshot_json"] or "{}")
        candidate_ids = [int(value) for value in snapshot.get("candidate_case_ids", [])]
        cohort_case_count = int(snapshot.get("candidate_count", len(candidate_ids)))
        untyped_case_count = int(snapshot.get("untyped_case_count", 0))
        connection.execute("UPDATE model_versions SET status = 'running' WHERE version_name = ?", (version_name,))
        connection.commit()
        records_by_case, usable_ids = _human_records(connection, candidate_ids, version_name)

    human_records = [record for case_records in records_by_case.values() for record in case_records]
    human_counts = Counter(record.label_index for record in human_records)
    observed_labels = sorted(human_counts)
    if len(human_records) < 40 or len(observed_labels) < 2:
        metrics = {
            "rejected_or_complex_cohort_cases": float(cohort_case_count),
            "untyped_complex_cases": float(untyped_case_count),
            "eligible_human_cases": float(len(usable_ids)),
            "eligible_human_roi_samples": float(len(human_records)),
            "observed_human_classes": float(len(observed_labels)),
        }
        with connect() as connection:
            _mark_version(
                connection,
                version_name,
                "validation_required",
                metrics,
                "O lote não tem diversidade supervisionada suficiente: são necessários pelo menos 40 recortes humanos e duas classes observadas. Não foi criado nem promovido qualquer modelo.",
                {"usable_case_ids": usable_ids, "human_class_counts": dict(human_counts)},
            )
        return

    human_train, human_holdout, train_case_ids, holdout_case_ids = _case_split(records_by_case)
    if not human_train or not human_holdout:
        with connect() as connection:
            _mark_version(
                connection,
                version_name,
                "validation_required",
                {"eligible_human_cases": float(len(usable_ids)), "eligible_human_roi_samples": float(len(human_records))},
                "Não foi possível reservar um conjunto humano de validação independente por caso. O candidato não foi treinado.",
                {"usable_case_ids": usable_ids},
            )
        return

    replay = _replay_records()
    locked_validation = _records_for_conditions("validation", DEFAULT_VALIDATION_CONDITIONS, use_existing_cache=True)
    _prepare_cache(human_train + human_holdout, rebuild=False)
    _prepare_cache(replay, rebuild=False)
    _prepare_cache(locked_validation, rebuild=False)

    train_records = human_train + replay
    class_counts = Counter(record.label_index for record in train_records)
    weights = torch.DoubleTensor([1.0 / class_counts[record.label_index] for record in train_records])
    sampler = WeightedRandomSampler(weights, num_samples=len(train_records), replacement=True)
    train_loader = DataLoader(CachedRoiDataset(train_records, augment=True), batch_size=16, sampler=sampler, num_workers=0)
    human_holdout_loader = DataLoader(CachedRoiDataset(human_holdout, augment=False), batch_size=16, shuffle=False, num_workers=0)
    locked_loader = DataLoader(CachedRoiDataset(locked_validation, augment=False), batch_size=16, shuffle=False, num_workers=0)

    active_weights = MODELS_DIR / "defect_type_resnet18.pt"
    if not active_weights.exists():
        raise RuntimeError("Pesos ativos do classificador de tipo ausentes.")
    checkpoint = torch.load(active_weights, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = DefectTypeResNet18()
    model.load_state_dict(state_dict)
    backbone_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": 3e-5},
            {"params": model.fc.parameters(), "lr": 2e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.04)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        model.train()
        for images, targets in train_loader:
            loss = loss_fn(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
        scheduler.step()
        human_metrics, _confusion, _logits, _targets = _evaluate(model, human_holdout_loader, loss_fn)
        score = _observed_macro_f1(human_metrics, human_holdout)
        if score > best_score:
            best_state = copy.deepcopy(model.state_dict())
            best_score = score
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("O fine-tuning não produziu um estado de modelo válido.")
    model.load_state_dict(best_state)
    human_metrics, human_confusion, _human_logits, _human_targets = _evaluate(model, human_holdout_loader, loss_fn)
    locked_metrics, locked_confusion, locked_logits, locked_targets = _evaluate(model, locked_loader, loss_fn)
    temperature = _fit_temperature(locked_logits, locked_targets)

    baseline_metadata = _load_active_metadata()
    baseline_metrics = baseline_metadata.get("metrics", {})
    baseline_macro_f1 = float(baseline_metrics.get("macro_f1", 0.0))
    baseline_minimum_recall = float(baseline_metrics.get("minimum_class_recall", 0.0))
    regression_gate_passed = (
        locked_metrics["macro_f1"] >= baseline_macro_f1 - REGRESSION_MACRO_F1_TOLERANCE
        and locked_metrics["minimum_class_recall"] >= baseline_minimum_recall - REGRESSION_RECALL_TOLERANCE
    )

    metrics: dict[str, Any] = {
        "rejected_or_complex_cohort_cases": cohort_case_count,
        "untyped_complex_cases": untyped_case_count,
        "eligible_human_cases": len(usable_ids),
        "eligible_human_roi_samples": len(human_records),
        "human_training_cases": len(train_case_ids),
        "human_holdout_cases": len(holdout_case_ids),
        "human_training_roi_samples": len(human_train),
        "human_holdout_roi_samples": len(human_holdout),
        "replay_roi_samples": len(replay),
        "locked_regression_roi_samples": len(locked_validation),
        "best_epoch": best_epoch,
        "human_holdout_observed_macro_f1": round(best_score, 4),
        "baseline_locked_macro_f1": baseline_macro_f1,
        "baseline_locked_minimum_class_recall": baseline_minimum_recall,
        "locked_macro_f1_delta": round(locked_metrics["macro_f1"] - baseline_macro_f1, 4),
        "locked_minimum_recall_delta": round(locked_metrics["minimum_class_recall"] - baseline_minimum_recall, 4),
        "locked_regression_gate_passed": float(regression_gate_passed),
        "temperature": temperature,
        **_flat_metrics("human_holdout_", human_metrics),
        **_flat_metrics("locked_", locked_metrics),
    }
    metadata = {
        "version_name": version_name,
        "model_type": "roi_rejected_case_candidate",
        "base_model": baseline_metadata.get("version_name"),
        "class_names": list(ROI_CLASS_NAMES),
        "training_target": "secondary ROI defect-type classifier only; approved normal images are excluded",
        "human_cases": {
            "total": len(usable_ids),
            "train": len(train_case_ids),
            "holdout": len(holdout_case_ids),
            "class_counts": {ROI_CLASS_NAMES[index]: count for index, count in sorted(human_counts.items())},
        },
        "replay_anchor": {
            "dataset": "AeBAD-S train conditions",
            "per_class_cap": REPLAY_PER_CLASS,
            "roi_samples": len(replay),
        },
        "validation": {
            "human_holdout": "20% of reviewed cases, split by inspection before fine-tuning",
            "locked_regression": "AeBAD-S view condition; development regression gate, not an independent aircraft test",
            "locked_roi_samples": len(locked_validation),
            "human_confusion_matrix": human_confusion,
            "locked_confusion_matrix": locked_confusion,
        },
        "metrics": metrics,
        "promotion": {
            "automatic": False,
            "locked_regression_gate_passed": regression_gate_passed,
            "required_action": "manual review and independent test before any operational promotion",
        },
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{version_name}.pt"
    metadata_path = MODELS_DIR / f"{version_name}.json"
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, checkpoint_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = (
        "Fine-tuning de candidato do classificador secundário de tipos de defeito, com caixas humanas de "
        "casos reprovados/complexos, replay fixo e validação separada por peça. "
    )
    notes += (
        "Passou a porta de regressão bloqueada; exige revisão humana e teste independente antes de promoção."
        if regression_gate_passed
        else "Não passou a porta de regressão bloqueada; não pode ser promovido."
    )
    with connect() as connection:
        _mark_version(
            connection,
            version_name,
            "validation_required",
            metrics,
            notes,
            {
                "usable_case_ids": usable_ids,
                "human_training_case_ids": train_case_ids,
                "human_holdout_case_ids": holdout_case_ids,
                "human_class_counts": {ROI_CLASS_NAMES[index]: count for index, count in sorted(human_counts.items())},
                "checkpoint": checkpoint_path.name,
                "metadata": metadata_path.name,
            },
        )
        for inspection_id in usable_ids:
            append_audit_event(
                connection,
                inspection_id,
                "local_training",
                "active_learning_secondary_rejected_candidate_completed",
                {
                    "version_name": version_name,
                    "locked_regression_gate_passed": regression_gate_passed,
                    "checkpoint": checkpoint_path.name,
                },
            )
        connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria um candidato local do classificador de tipo a partir de revisões humanas.")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cv2.setNumThreads(0)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    try:
        _run(args.version)
    except Exception as error:
        init_db()
        with connect() as connection:
            _mark_version(
                connection,
                args.version,
                "validation_required",
                {"error": str(error)},
                "O candidato de aprendizagem ativa falhou antes de poder ser validado; o modelo em produção não foi alterado.",
            )
        raise


if __name__ == "__main__":
    main()
