from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import DATASETS_DIR, RUNS_DIR, UPLOADS_DIR, WORKSPACE_ROOT, ensure_directories
from .db import (
    append_audit_event,
    connect,
    create_annotation_revision,
    init_db,
    now_iso,
    serialize_inspection,
)
from .schemas import BatchDecisionPayload, CandidatePayload, ReviewPayload
from .services.image_analysis import analyse_image, save_xai_overlay
from .services.seed import seed_dlr_demo
from .services.training import (
    launch_active_learning_candidate,
    queue_ready_active_learning_candidates,
    training_queue_status,
)


ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ATTRIBUTIONS_PATH = WORKSPACE_ROOT / "docs" / "ATTRIBUTIONS.md"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_directories()
    init_db()
    yield


app = FastAPI(
    title="AeroVision Inspector",
    version="0.1.0",
    description="Apoio local e auditável à inspeção visual de componentes metálicos.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    connection = connect()
    try:
        yield connection
    finally:
        connection.close()


def get_inspection_or_404(connection: sqlite3.Connection, inspection_id: int) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspeção não encontrada.")
    return row


def _path_is_owned(path: Path) -> bool:
    resolved = path.resolve()
    for root in (DATASETS_DIR, UPLOADS_DIR, RUNS_DIR):
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _serve_database_file(connection: sqlite3.Connection, inspection_id: int, column: str, media_type: str) -> FileResponse:
    row = get_inspection_or_404(connection, inspection_id)
    value = row[column]
    if not value:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ficheiro ainda não disponível.")
    path = Path(str(value))
    if not path.exists() or not _path_is_owned(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ficheiro de inspeção indisponível.")
    return FileResponse(path, media_type=media_type, filename=path.name)


def _serialize_many(connection: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[dict]:
    return [serialize_inspection(connection, row) for row in rows]


def _analyse_and_store(
    connection: sqlite3.Connection,
    inspection_id: int,
    *,
    only_if_pending: bool = False,
) -> dict:
    """Run local inference and persist it, without reprocessing completed items."""
    row = get_inspection_or_404(connection, inspection_id)
    if only_if_pending and row["ai_state"] != "not_analysed":
        return serialize_inspection(connection, row)

    try:
        result = analyse_image(row["source_path"])
        xai_path = save_xai_overlay(inspection_id, result.xai_bgr)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error

    connection.execute(
        """
        UPDATE inspections
        SET updated_at = ?, ai_state = ?, anomaly_score = ?, operational_confidence = ?,
            quality_state = ?, quality_json = ?, model_version = ?, model_boxes_json = ?, xai_path = ?
        WHERE id = ?
        """,
        (
            now_iso(),
            result.ai_state,
            result.anomaly_score,
            result.operational_confidence,
            result.quality_state,
            json.dumps(result.quality, ensure_ascii=False),
            result.model_version,
            json.dumps(result.model_boxes, ensure_ascii=False),
            xai_path,
            inspection_id,
        ),
    )
    append_audit_event(
        connection,
        inspection_id,
        "local_model",
        "analysis_completed",
        {
            "ai_state": result.ai_state,
            "anomaly_score": result.anomaly_score,
            "operational_confidence": result.operational_confidence,
            "quality_state": result.quality_state,
            "model_version": result.model_version,
            "model_box_count": len(result.model_boxes),
            "xai_method": result.quality.get("xai_method"),
        },
    )
    connection.commit()
    return serialize_inspection(connection, get_inspection_or_404(connection, inspection_id))


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "AeroVision Inspector API", "mode": "local/offline"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "inference": "local-only", "external_apis": "none"}


@app.get("/api/attributions")
def attributions() -> dict[str, str]:
    return {"markdown": ATTRIBUTIONS_PATH.read_text(encoding="utf-8") if ATTRIBUTIONS_PATH.exists() else ""}


@app.get("/api/dashboard")
def dashboard(connection: sqlite3.Connection = Depends(get_db)) -> dict:
    counts = {
        "total": int(connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]),
        "analysed": int(
            connection.execute("SELECT COUNT(*) FROM inspections WHERE ai_state NOT IN ('not_analysed', 'model_not_ready')").fetchone()[0]
        ),
        "awaiting_human_decision": int(
            connection.execute("SELECT COUNT(*) FROM inspections WHERE final_decision IS NULL").fetchone()[0]
        ),
        "human_decided": int(
            connection.execute("SELECT COUNT(*) FROM inspections WHERE final_decision IS NOT NULL").fetchone()[0]
        ),
        "likely_anomaly": int(
            connection.execute("SELECT COUNT(*) FROM inspections WHERE ai_state = 'likely_anomaly'").fetchone()[0]
        ),
        "ambiguous": int(
            connection.execute("SELECT COUNT(*) FROM inspections WHERE ai_state = 'ambiguous'").fetchone()[0]
        ),
        "quality_warning": int(
            connection.execute("SELECT COUNT(*) FROM inspections WHERE quality_state = 'capture_quality_warning'").fetchone()[0]
        ),
    }
    latest_rows = connection.execute(
        "SELECT * FROM inspections ORDER BY final_decision IS NULL DESC, updated_at DESC LIMIT 8"
    ).fetchall()
    return {"counts": counts, "latest": _serialize_many(connection, latest_rows), "training": training_queue_status(connection)}


@app.post("/api/inspections/seed")
def seed_demo(connection: sqlite3.Connection = Depends(get_db)) -> dict:
    result = seed_dlr_demo(connection)
    connection.commit()
    if result["created"]:
        message = (
            f"Foram registadas {result['created']} imagens DLR de validação, equilibradas por normalidade/amolgadela, "
            f"de um conjunto local de {result['available']}; nenhuma foi analisada automaticamente. A etiqueta de origem não é usada pela IA."
        )
    else:
        message = (
            f"As {result['available']} imagens DLR de demonstração já estão na fila local; nenhuma foi analisada "
            "automaticamente. A etiqueta de origem não é usada pela IA."
        )
    return {
        **result,
        "message": message,
    }


@app.post("/api/inspections/upload", status_code=status.HTTP_201_CREATED)
async def upload_inspection(
    image: UploadFile = File(...),
    connection: sqlite3.Connection = Depends(get_db),
) -> dict:
    original_name = Path(image.filename or "inspection.png").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Envie uma imagem PNG, JPEG, BMP ou TIFF.")
    destination = UPLOADS_DIR / f"{uuid4().hex}{suffix}"
    try:
        with destination.open("wb") as target:
            shutil.copyfileobj(image.file, target)
    finally:
        await image.close()
    if cv2.imread(str(destination), cv2.IMREAD_COLOR) is None:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="O ficheiro não contém uma imagem válida.")

    created_at = now_iso()
    cursor = connection.execute(
        """
        INSERT INTO inspections (
            created_at, updated_at, source_name, source_path, source_label,
            part_profile, acquisition_context_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            created_at,
            original_name,
            str(destination.resolve()),
            None,
            "aero_engine_blade_rgb",
            json.dumps({"dataset": "user_upload", "capture_mode": "local upload"}, ensure_ascii=False),
        ),
    )
    inspection_id = int(cursor.lastrowid)
    append_audit_event(
        connection,
        inspection_id,
        "operator",
        "inspection_uploaded",
        {"original_name": original_name, "stored_name": destination.name},
    )
    connection.commit()
    # A newly added image is analysed immediately. The guard also makes this
    # safe if an upload workflow retries after the record was already analysed.
    return _analyse_and_store(connection, inspection_id, only_if_pending=True)


@app.get("/api/inspections")
def list_inspections(
    limit: int = 100,
    connection: sqlite3.Connection = Depends(get_db),
) -> dict:
    safe_limit = max(1, min(limit, 500))
    rows = connection.execute(
        """
        SELECT * FROM inspections
        ORDER BY final_decision IS NULL DESC, updated_at DESC
        LIMIT ?
        """,
        (safe_limit,),
    ).fetchall()
    return {"items": _serialize_many(connection, rows)}


@app.get("/api/inspections/{inspection_id}")
def get_inspection(inspection_id: int, connection: sqlite3.Connection = Depends(get_db)) -> dict:
    return serialize_inspection(connection, get_inspection_or_404(connection, inspection_id))


@app.get("/api/inspections/{inspection_id}/image")
def inspection_image(inspection_id: int, connection: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    return _serve_database_file(connection, inspection_id, "source_path", "image/*")


@app.get("/api/inspections/{inspection_id}/xai")
def inspection_xai(inspection_id: int, connection: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    return _serve_database_file(connection, inspection_id, "xai_path", "image/jpeg")


@app.post("/api/inspections/{inspection_id}/analyse")
def analyse_inspection(inspection_id: int, connection: sqlite3.Connection = Depends(get_db)) -> dict:
    return _analyse_and_store(connection, inspection_id)


@app.post("/api/inspections/{inspection_id}/review")
def review_inspection(
    inspection_id: int,
    payload: ReviewPayload,
    background_tasks: BackgroundTasks,
    connection: sqlite3.Connection = Depends(get_db),
) -> dict:
    get_inspection_or_404(connection, inspection_id)
    boxes = [box.model_dump() for box in payload.boxes]
    revision_id = create_annotation_revision(connection, inspection_id, payload.actor, payload.reason or None, boxes)
    reviewed_at = now_iso()
    connection.execute(
        """
        UPDATE inspections
        SET updated_at = ?, final_decision = ?, final_reason = ?, final_actor = ?, final_at = ?
        WHERE id = ?
        """,
        (reviewed_at, payload.decision, payload.reason, payload.actor, reviewed_at, inspection_id),
    )
    append_audit_event(
        connection,
        inspection_id,
        payload.actor,
        "human_review_completed",
        {
            "decision": payload.decision,
            "reason": payload.reason,
            "annotation_revision_id": revision_id,
            "box_count": len(boxes),
        },
    )
    candidates = queue_ready_active_learning_candidates(connection, payload.actor)
    connection.commit()
    for candidate in candidates:
        background_tasks.add_task(launch_active_learning_candidate, candidate)
    return {
        "inspection": serialize_inspection(connection, get_inspection_or_404(connection, inspection_id)),
        "training_trigger": candidates[0] if candidates else None,
        "training_triggers": candidates,
    }


@app.post("/api/inspections/batch-decision")
def batch_decision(
    payload: BatchDecisionPayload,
    background_tasks: BackgroundTasks,
    connection: sqlite3.Connection = Depends(get_db),
) -> dict:
    unique_ids = list(dict.fromkeys(payload.inspection_ids))
    rows = connection.execute(
        f"SELECT id FROM inspections WHERE id IN ({','.join('?' for _ in unique_ids)})", unique_ids
    ).fetchall()
    found_ids = {int(row["id"]) for row in rows}
    reviewed_at = now_iso()
    for inspection_id in found_ids:
        connection.execute(
            """
            UPDATE inspections
            SET updated_at = ?, final_decision = ?, final_reason = ?, final_actor = ?, final_at = ?
            WHERE id = ?
            """,
            (reviewed_at, payload.decision, payload.reason, payload.actor, reviewed_at, inspection_id),
        )
        append_audit_event(
            connection,
            inspection_id,
            payload.actor,
            "batch_human_decision_completed",
            {
                "decision": payload.decision,
                "reason": payload.reason,
                "batch_size": len(found_ids),
                "annotations_changed": False,
            },
        )
    candidates = queue_ready_active_learning_candidates(connection, payload.actor)
    connection.commit()
    for candidate in candidates:
        background_tasks.add_task(launch_active_learning_candidate, candidate)
    return {
        "updated_ids": sorted(found_ids),
        "missing_ids": sorted(set(unique_ids) - found_ids),
        "training_trigger": candidates[0] if candidates else None,
        "training_triggers": candidates,
        "message": "A decisão em lote não alterou as caixas de delimitação individuais.",
    }


@app.get("/api/inspections/{inspection_id}/history")
def inspection_history(inspection_id: int, connection: sqlite3.Connection = Depends(get_db)) -> dict:
    get_inspection_or_404(connection, inspection_id)
    events = connection.execute(
        """
        SELECT id, created_at, actor, event_type, payload_json, previous_hash, event_hash
        FROM audit_events WHERE inspection_id = ? ORDER BY id ASC
        """,
        (inspection_id,),
    ).fetchall()
    revisions = connection.execute(
        """
        SELECT id, revision_number, created_at, actor, note
        FROM annotation_revisions WHERE inspection_id = ? ORDER BY revision_number DESC
        """,
        (inspection_id,),
    ).fetchall()
    revision_data: list[dict] = []
    for revision in revisions:
        item = dict(revision)
        item["boxes"] = [
            dict(box)
            for box in connection.execute("SELECT * FROM bounding_boxes WHERE revision_id = ? ORDER BY id", (revision["id"],)).fetchall()
        ]
        revision_data.append(item)
    return {
        "events": [{**dict(event), "payload": json.loads(event["payload_json"])} for event in events],
        "annotation_revisions": revision_data,
    }


@app.get("/api/training/queue")
def training_queue(connection: sqlite3.Connection = Depends(get_db)) -> dict:
    return training_queue_status(connection)


@app.post("/api/training/trigger")
def trigger_training(
    payload: CandidatePayload,
    background_tasks: BackgroundTasks,
    connection: sqlite3.Connection = Depends(get_db),
) -> dict:
    candidates = queue_ready_active_learning_candidates(connection, payload.actor)
    connection.commit()
    if not candidates:
        return {"triggered": False, "queue": training_queue_status(connection)}
    for candidate in candidates:
        background_tasks.add_task(launch_active_learning_candidate, candidate)
    return {
        "triggered": True,
        "candidate": candidates[0],
        "candidates": candidates,
        "queue": training_queue_status(connection),
    }
