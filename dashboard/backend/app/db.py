from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from typing import Any, Iterable

from .config import DATABASE_PATH, ensure_directories


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    ensure_directories()
    # FastAPI may resolve a synchronous dependency and execute its endpoint in
    # different worker threads. Each request still owns its own connection, but
    # SQLite must permit that hand-off within the request lifecycle.
    connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_path TEXT NOT NULL UNIQUE,
                source_label TEXT,
                part_profile TEXT NOT NULL DEFAULT 'aero_engine_blade_rgb',
                acquisition_context_json TEXT NOT NULL DEFAULT '{}',
                ai_state TEXT NOT NULL DEFAULT 'not_analysed',
                anomaly_score REAL,
                operational_confidence REAL,
                quality_state TEXT NOT NULL DEFAULT 'not_assessed',
                quality_json TEXT NOT NULL DEFAULT '{}',
                model_version TEXT,
                model_boxes_json TEXT NOT NULL DEFAULT '[]',
                xai_path TEXT,
                final_decision TEXT,
                final_reason TEXT,
                final_actor TEXT,
                final_at TEXT
            );

            CREATE TABLE IF NOT EXISTS annotation_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
                revision_number INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                note TEXT,
                UNIQUE(inspection_id, revision_number)
            );

            CREATE TABLE IF NOT EXISTS bounding_boxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                revision_id INTEGER NOT NULL REFERENCES annotation_revisions(id) ON DELETE CASCADE,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                label TEXT NOT NULL DEFAULT 'defect',
                confidence REAL,
                source TEXT NOT NULL DEFAULT 'human'
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER REFERENCES inspections(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT,
                event_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_name TEXT NOT NULL UNIQUE,
                model_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                dataset_snapshot_json TEXT NOT NULL DEFAULT '{}',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT
            );

            CREATE INDEX IF NOT EXISTS ix_inspections_created_at ON inspections(created_at DESC);
            CREATE INDEX IF NOT EXISTS ix_audit_inspection ON audit_events(inspection_id, id DESC);
            CREATE INDEX IF NOT EXISTS ix_revisions_inspection ON annotation_revisions(inspection_id, revision_number DESC);
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def append_audit_event(
    connection: sqlite3.Connection,
    inspection_id: int | None,
    actor: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    previous = connection.execute(
        "SELECT event_hash FROM audit_events WHERE inspection_id IS ? ORDER BY id DESC LIMIT 1",
        (inspection_id,),
    ).fetchone()
    previous_hash = previous["event_hash"] if previous else ""
    created_at = now_iso()
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    material = "|".join((previous_hash, str(inspection_id), created_at, actor, event_type, payload_json))
    event_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()
    connection.execute(
        """
        INSERT INTO audit_events (
            inspection_id, created_at, actor, event_type, payload_json, previous_hash, event_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (inspection_id, created_at, actor, event_type, payload_json, previous_hash or None, event_hash),
    )


def create_annotation_revision(
    connection: sqlite3.Connection,
    inspection_id: int,
    actor: str,
    note: str | None,
    boxes: Iterable[dict[str, Any]],
) -> int:
    latest = connection.execute(
        "SELECT COALESCE(MAX(revision_number), 0) AS max_revision FROM annotation_revisions WHERE inspection_id = ?",
        (inspection_id,),
    ).fetchone()
    revision_number = int(latest["max_revision"]) + 1
    cursor = connection.execute(
        """
        INSERT INTO annotation_revisions (inspection_id, revision_number, created_at, actor, note)
        VALUES (?, ?, ?, ?, ?)
        """,
        (inspection_id, revision_number, now_iso(), actor, note),
    )
    revision_id = int(cursor.lastrowid)
    for box in boxes:
        connection.execute(
            """
            INSERT INTO bounding_boxes (revision_id, x, y, width, height, label, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                float(box["x"]),
                float(box["y"]),
                float(box["width"]),
                float(box["height"]),
                str(box.get("label") or "defect"),
                box.get("confidence"),
                str(box.get("source") or "human"),
            ),
        )
    return revision_id


def get_latest_annotation_boxes(connection: sqlite3.Connection, inspection_id: int) -> list[dict[str, Any]]:
    revision = connection.execute(
        """
        SELECT id, revision_number, created_at, actor, note
        FROM annotation_revisions
        WHERE inspection_id = ?
        ORDER BY revision_number DESC
        LIMIT 1
        """,
        (inspection_id,),
    ).fetchone()
    if not revision:
        return []
    boxes = connection.execute(
        "SELECT * FROM bounding_boxes WHERE revision_id = ? ORDER BY id",
        (revision["id"],),
    ).fetchall()
    return [dict(row) for row in boxes]


def serialize_inspection(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["acquisition_context"] = json_loads(result.pop("acquisition_context_json"), {})
    result["quality"] = json_loads(result.pop("quality_json"), {})
    result["model_boxes"] = json_loads(result.pop("model_boxes_json"), [])
    result["human_boxes"] = get_latest_annotation_boxes(connection, int(result["id"]))
    result["image_url"] = f"/api/inspections/{result['id']}/image"
    result["xai_url"] = f"/api/inspections/{result['id']}/xai" if result.get("xai_path") else None
    return result

