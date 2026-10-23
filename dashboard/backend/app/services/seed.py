from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from ..config import DATASETS_DIR, RUNS_DIR
from ..db import append_audit_event, now_iso


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEMO_COUNT_PER_GROUP = 20


def _evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    positions = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[position] for position in positions]


def _stratified_demo_selection(items: list[Path], count: int) -> list[Path]:
    """Keep the expanded pitch queue balanced across capture conditions.

    AeBAD-S groups each label by `background`, `illumination`, `same` and
    `view`. Selecting separately from each condition means the demonstration
    does not accidentally look good only under one lighting arrangement.
    """
    if len(items) <= count:
        return items
    by_condition: dict[str, list[Path]] = {}
    for path in items:
        by_condition.setdefault(path.parent.name, []).append(path)
    conditions = sorted(by_condition)
    base, remainder = divmod(count, len(conditions))
    selected: list[Path] = []
    for index, condition in enumerate(conditions):
        selected.extend(_evenly_spaced(by_condition[condition], base + int(index < remainder)))
    selected = list(dict.fromkeys(selected))
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(_evenly_spaced([item for item in items if item not in selected_set], count - len(selected)))
    return sorted(selected)


def seed_aebad_demo(connection: sqlite3.Connection, count_per_group: int = DEMO_COUNT_PER_GROUP) -> dict[str, int]:
    """Register local AeBAD images as inspection records without leaking labels to AI.

    `source_label` is provenance metadata. It is never passed to the model or used
    to manufacture an analysis result.
    """
    root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "test"
    if not root.exists():
        return {"created": 0, "available": 0}

    created = 0
    available = 0
    for source_label in ("good", "ablation", "breakdown", "fracture", "groove"):
        label_root = root / source_label
        candidates = sorted(
            path
            for path in label_root.rglob("*")
            if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
        )
        selected = _stratified_demo_selection(candidates, count_per_group) if candidates else []
        available += len(selected)
        for path in selected:
            created_at = now_iso()
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO inspections (
                    created_at, updated_at, source_name, source_path, source_label,
                    part_profile, acquisition_context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    created_at,
                    f"AeBAD-S · {path.parent.name} · {path.name}",
                    str(path.resolve()),
                    source_label,
                    "aero_engine_blade_rgb",
                    json.dumps(
                        {
                            "dataset": "AeBAD-S",
                            "domain_variant": path.parent.name,
                            "capture_mode": "public reference image",
                            "demo_cohort": f"{count_per_group}_per_source_label",
                            "lighting_control": "not available in this offline demo",
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            if cursor.rowcount:
                inspection_id = int(cursor.lastrowid)
                append_audit_event(
                    connection,
                    inspection_id,
                    "system",
                    "inspection_seeded",
                    {
                        "source_dataset": "AeBAD-S",
                        "source_path": str(path.resolve()),
                        "provenance_label_retained": True,
                    },
                )
                created += 1
    return {"created": created, "available": available}


def seed_dlr_demo(connection: sqlite3.Connection, count_per_group: int = DEMO_COUNT_PER_GROUP) -> dict[str, int]:
    """Register a balanced DLR validation cohort for the RT-DETR dashboard demo.

    The selected files are from the official DLR validation split, never from
    the sealed test set used to report the detector result. Provenance labels
    stay in the database for a later human check but are never supplied to the
    inference function.
    """
    manifest_path = RUNS_DIR / "dlr_benchmark_20261022T152053Z" / "protocol_manifest.csv"
    if not manifest_path.exists():
        return {"created": 0, "available": 0}
    by_label: dict[str, list[Path]] = {"normal": [], "dent": []}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("ground_truth") or "").strip().lower()
            if row.get("official_split") != "val" or label not in by_label:
                continue
            path = Path(str(row.get("image_path") or ""))
            if path.is_file():
                by_label[label].append(path)

    created = 0
    available = 0
    for label, candidates in by_label.items():
        selected = _evenly_spaced(sorted(set(candidates)), count_per_group)
        available += len(selected)
        for path in selected:
            created_at = now_iso()
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO inspections (
                    created_at, updated_at, source_name, source_path, source_label,
                    part_profile, acquisition_context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    created_at,
                    f"DLR Aircraft Dent · validação · {label} · {path.name}",
                    str(path.resolve()),
                    label,
                    "aircraft_metal_surface_rgb",
                    json.dumps(
                        {
                            "dataset": "DLR Aircraft Dent",
                            "official_split": "validation",
                            "capture_mode": "public reference image",
                            "demo_cohort": f"{count_per_group}_per_ground_truth",
                            "note": "A etiqueta de proveniência não é fornecida ao modelo durante a inferência.",
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            if cursor.rowcount:
                inspection_id = int(cursor.lastrowid)
                append_audit_event(
                    connection,
                    inspection_id,
                    "system",
                    "inspection_seeded",
                    {
                        "source_dataset": "DLR Aircraft Dent",
                        "official_split": "validation",
                        "source_path": str(path.resolve()),
                        "provenance_label_retained": True,
                    },
                )
                created += 1
    return {"created": created, "available": available}
