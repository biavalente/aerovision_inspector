from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ..config import WORKSPACE_ROOT
from ..db import append_audit_event, now_iso
from ..defect_labels import TRAINABLE_DEFECT_LABELS, canonical_defect_label


# Both loops intentionally count reviewed *inspections*, not bounding boxes.
# It makes every training batch traceable to exactly 250 final human decisions.
RETRAIN_THRESHOLD = 250
PRIMARY_NORMAL_MODEL_TYPE = "autoencoder_normal_adaptation_candidate"
SECONDARY_REJECTED_MODEL_TYPE = "roi_rejected_case_candidate"


def _latest_box_labels(connection: sqlite3.Connection, inspection_id: int) -> list[str]:
    rows = connection.execute(
        """
        SELECT b.label
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
    return [str(row["label"]) for row in rows]


def _final_decision_rows(connection: sqlite3.Connection, decisions: tuple[str, ...]) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in decisions)
    rows = connection.execute(
        f"""
        SELECT id, source_path, final_decision, final_at
        FROM inspections
        WHERE final_decision IN ({placeholders})
        ORDER BY final_at ASC, id ASC
        """,
        decisions,
    ).fetchall()
    # A queue position must correspond to an image the local trainer can still
    # read. This avoids a nominal 250-case candidate failing because uploads
    # were later removed from local storage.
    return [row for row in rows if Path(str(row["source_path"])).is_file()]


def _approved_normal_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """All final human approvals teach the normal-only primary model.

    The model still receives no defect image in this loop. A camera-quality
    warning is not an exclusion: if an inspector ultimately approves the
    image as representative of acceptable material, that final decision is
    the ground truth requested by the operating team.
    """
    return _final_decision_rows(connection, ("approved_visual",))


def _rejected_or_complex_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Cases rejected by the inspector, plus cases escalated as complex.

    They form the 250-case cohort for the secondary defect-type candidate.
    A complex case may lack a settled type label; it is retained for audit and
    later resolution but is not silently converted into a supervised class.
    """
    return _final_decision_rows(connection, ("rejected_visual", "escalated_ndt"))


def _processed_case_count(connection: sqlite3.Connection, model_type: str) -> int:
    rows = connection.execute(
        """
        SELECT dataset_snapshot_json
        FROM model_versions
        WHERE model_type = ?
          AND status IN ('queued', 'running', 'completed', 'validation_required')
        """,
        (model_type,),
    ).fetchall()
    counts: list[int] = []
    for row in rows:
        try:
            counts.append(int(json.loads(row["dataset_snapshot_json"]).get("processed_case_count", 0)))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return max(counts, default=0)


def _queue_summary(
    connection: sqlite3.Connection,
    *,
    rows: list[sqlite3.Row],
    model_type: str,
    target: str,
    rule: str,
) -> dict[str, Any]:
    processed = _processed_case_count(connection, model_type)
    pending = max(0, len(rows) - processed)
    return {
        "threshold": RETRAIN_THRESHOLD,
        "reviewed_cases": len(rows),
        "processed_cases": processed,
        "pending_cases": pending,
        "remaining_to_trigger": max(0, RETRAIN_THRESHOLD - pending),
        "eligible_now": pending >= RETRAIN_THRESHOLD,
        "training_target": target,
        "eligibility_rule": rule,
    }


def _recent_versions(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT version_name, model_type, status, created_at, dataset_snapshot_json, metrics_json, notes
        FROM model_versions
        ORDER BY id DESC
        LIMIT 12
        """
    ).fetchall()
    versions: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["dataset_snapshot"] = json.loads(item.pop("dataset_snapshot_json") or "{}")
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        versions.append(item)
    return versions


def training_queue_status(connection: sqlite3.Connection) -> dict[str, Any]:
    normal_rows = _approved_normal_rows(connection)
    rejected_rows = _rejected_or_complex_rows(connection)
    primary = _queue_summary(
        connection,
        rows=normal_rows,
        model_type=PRIMARY_NORMAL_MODEL_TYPE,
        target="modelo primário de normalidade (autoencoder)",
        rule="apenas imagens com decisão final humana 'sem defeito'",
    )
    secondary = _queue_summary(
        connection,
        rows=rejected_rows,
        model_type=SECONDARY_REJECTED_MODEL_TYPE,
        target="classificador secundário de tipos de defeito por região",
        rule="decisão final humana 'reprovada' ou 'caso complexo'; só caixas com tipo explícito ensinam uma classe",
    )
    return {
        "threshold": RETRAIN_THRESHOLD,
        "primary_normal_learning": primary,
        "secondary_rejected_learning": secondary,
        # Compatibility fields for an older dashboard build. They point to
        # the secondary queue because it replaced the original single loop.
        "training_target": secondary["training_target"],
        "reviewed_complex_cases": secondary["reviewed_cases"],
        "processed_complex_cases": secondary["processed_cases"],
        "pending_complex_cases": secondary["pending_cases"],
        "remaining_to_trigger": secondary["remaining_to_trigger"],
        "eligible_now": primary["eligible_now"] or secondary["eligible_now"],
        "model_versions": _recent_versions(connection),
    }


def _queue_candidate(
    connection: sqlite3.Connection,
    actor: str,
    *,
    rows: list[sqlite3.Row],
    model_type: str,
    version_prefix: str,
    selection_rule: str,
    training_target: str,
    event_type: str,
    snapshot_extra: Callable[[list[sqlite3.Row]], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    processed = _processed_case_count(connection, model_type)
    if len(rows) - processed < RETRAIN_THRESHOLD:
        return None

    selected = rows[processed : processed + RETRAIN_THRESHOLD]
    version_name = f"{version_prefix}_{now_iso().replace(':', '').replace('+00:00', 'Z')}"
    snapshot: dict[str, Any] = {
        "candidate_case_ids": [int(row["id"]) for row in selected],
        "candidate_count": len(selected),
        "processed_case_count": processed + len(selected),
        "selection_rule": selection_rule,
        "training_target": training_target,
        "decisions": {str(row["id"]): str(row["final_decision"]) for row in selected},
    }
    if snapshot_extra:
        snapshot.update(snapshot_extra(selected))
    connection.execute(
        """
        INSERT INTO model_versions (
            version_name, model_type, status, created_at, dataset_snapshot_json, metrics_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_name,
            model_type,
            "queued",
            now_iso(),
            json.dumps(snapshot, ensure_ascii=False),
            "{}",
            "Candidato de aprendizagem contínua; nunca é promovido automaticamente.",
        ),
    )
    for row in selected:
        append_audit_event(
            connection,
            int(row["id"]),
            actor,
            event_type,
            {"version_name": version_name, "candidate_count": len(selected), "model_type": model_type},
        )
    return {
        "version_name": version_name,
        "candidate_count": len(selected),
        "model_type": model_type,
        "training_target": training_target,
    }


def _secondary_snapshot_extra(connection: sqlite3.Connection) -> Callable[[list[sqlite3.Row]], dict[str, Any]]:
    def build(rows: list[sqlite3.Row]) -> dict[str, Any]:
        supervised_case_ids: list[int] = []
        complex_without_type_ids: list[int] = []
        for row in rows:
            inspection_id = int(row["id"])
            labels = {canonical_defect_label(label) for label in _latest_box_labels(connection, inspection_id)}
            if labels.intersection(TRAINABLE_DEFECT_LABELS):
                supervised_case_ids.append(inspection_id)
            else:
                complex_without_type_ids.append(inspection_id)
        return {
            "supervised_type_case_ids": supervised_case_ids,
            "complex_or_untyped_case_ids": complex_without_type_ids,
            "supervised_type_case_count": len(supervised_case_ids),
            "untyped_case_count": len(complex_without_type_ids),
        }

    return build


def queue_ready_active_learning_candidates(connection: sqlite3.Connection, actor: str) -> list[dict[str, Any]]:
    """Queue every independent 250-decision batch that is ready now.

    The loops use separate counters: an approval can influence only the
    normality autoencoder, and a rejected/complex review only the secondary
    type-classifier candidate.
    """
    candidates: list[dict[str, Any]] = []
    primary = _queue_candidate(
        connection,
        actor,
        rows=_approved_normal_rows(connection),
        model_type=PRIMARY_NORMAL_MODEL_TYPE,
        version_prefix="autoencoder_normal_active_learning",
        selection_rule="250 imagens com decisão final humana aprovada como sem defeito",
        training_target="primary normality autoencoder only",
        event_type="active_learning_primary_normal_candidate_queued",
    )
    if primary:
        candidates.append(primary)
    secondary = _queue_candidate(
        connection,
        actor,
        rows=_rejected_or_complex_rows(connection),
        model_type=SECONDARY_REJECTED_MODEL_TYPE,
        version_prefix="roi_rejected_active_learning",
        selection_rule=(
            "250 imagens com decisão final humana reprovada ou encaminhada como caso complexo; "
            "apenas caixas humanas com tipo explícito entram na perda supervisionada"
        ),
        training_target="secondary ROI defect-type classifier only",
        event_type="active_learning_secondary_rejected_candidate_queued",
        snapshot_extra=_secondary_snapshot_extra(connection),
    )
    if secondary:
        candidates.append(secondary)
    return candidates


def launch_active_learning_candidate(candidate: dict[str, Any]) -> None:
    model_type = str(candidate["model_type"])
    script = {
        PRIMARY_NORMAL_MODEL_TYPE: WORKSPACE_ROOT
        / "exploratory_models"
        / "autoencoder"
        / "scripts"
        / "finetune_autoencoder_from_normals.py",
        SECONDARY_REJECTED_MODEL_TYPE: WORKSPACE_ROOT
        / "exploratory_models"
        / "resnet"
        / "scripts"
        / "finetune_from_reviews.py",
    }.get(model_type)
    if not script:
        return
    if not script.exists():
        return
    subprocess.Popen(
        [sys.executable, str(script), "--version", str(candidate["version_name"])],
        cwd=str(WORKSPACE_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
