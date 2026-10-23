from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"
MODELS_DIR = DATA_DIR / "models"
DATABASE_PATH = DATA_DIR / "aerovision.sqlite3"
DATASETS_DIR = WORKSPACE_ROOT / "datasets"


def ensure_directories() -> None:
    for directory in (DATA_DIR, UPLOADS_DIR, RUNS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

