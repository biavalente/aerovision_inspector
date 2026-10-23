from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATASETS_DIR, MODELS_DIR  # noqa: E402
from backend.app.db import append_audit_event, connect, init_db  # noqa: E402
from backend.app.models.autoencoder import BladeAutoencoder  # noqa: E402
from backend.app.services.image_analysis import MODEL_INPUT_SIZE  # noqa: E402


FINE_TUNE_EPOCHS = 5
HUMAN_HOLDOUT_FRACTION = 0.20
REPLAY_NORMAL_IMAGES = 250
PUBLIC_NORMAL_REGRESSION_CAP = 160


class NormalImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Imagem inacessível: {self.paths[index]}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div(255.0)


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


def _normal_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _case_split(case_paths: dict[int, Path]) -> tuple[dict[int, Path], dict[int, Path]]:
    case_ids = sorted(case_paths)
    random.Random(42).shuffle(case_ids)
    holdout_count = max(1, round(len(case_ids) * HUMAN_HOLDOUT_FRACTION)) if len(case_ids) > 1 else 0
    holdout_ids = set(case_ids[:holdout_count])
    train = {case_id: path for case_id, path in case_paths.items() if case_id not in holdout_ids}
    holdout = {case_id: path for case_id, path in case_paths.items() if case_id in holdout_ids}
    if not train and holdout:
        train, holdout = holdout, {}
    return train, holdout


def _raw_top_error(model: nn.Module, tensor: torch.Tensor) -> float:
    with torch.inference_mode():
        reconstruction = model(tensor.unsqueeze(0)).squeeze(0)
    error = torch.mean(torch.abs(tensor - reconstruction), dim=0).cpu().numpy()
    return float(np.percentile(error, 99.2))


def _score_paths(model: nn.Module, paths: list[Path]) -> list[float]:
    dataset = NormalImageDataset(paths)
    return [_raw_top_error(model, dataset[index]) for index in range(len(dataset))]


def _summary(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"count": 0.0}
    return {
        "count": float(len(scores)),
        "top_error_p50": round(float(np.percentile(scores, 50)), 6),
        "top_error_p95": round(float(np.percentile(scores, 95)), 6),
        "top_error_p99": round(float(np.percentile(scores, 99)), 6),
        "top_error_mean": round(float(np.mean(scores)), 6),
    }


def _load_active_metadata() -> dict[str, Any]:
    path = MODELS_DIR / "blade_autoencoder.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
        if not candidate_ids:
            raise RuntimeError("O candidato não contém imagens aprovadas.")
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = connection.execute(
            f"""
            SELECT id, source_path
            FROM inspections
            WHERE id IN ({placeholders}) AND final_decision = 'approved_visual'
            """,
            candidate_ids,
        ).fetchall()
        case_paths = {int(row["id"]): Path(str(row["source_path"])) for row in rows if Path(str(row["source_path"])).exists()}
        connection.execute("UPDATE model_versions SET status = 'running' WHERE version_name = ?", (version_name,))
        connection.commit()

    human_train_cases, human_holdout_cases = _case_split(case_paths)
    if len(human_train_cases) < 8 or not human_holdout_cases:
        metrics = {
            "approved_cohort_cases": len(candidate_ids),
            "usable_human_normal_cases": len(case_paths),
            "human_training_cases": len(human_train_cases),
            "human_holdout_cases": len(human_holdout_cases),
        }
        with connect() as connection:
            _mark_version(
                connection,
                version_name,
                "validation_required",
                metrics,
                "Não foi possível reservar imagens humanas normais suficientes para treino e validação separados. O modelo ativo não foi alterado.",
                {"usable_case_ids": sorted(case_paths)},
            )
        return

    replay_root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "train" / "good"
    replay_candidates = _normal_paths(replay_root)
    if len(replay_candidates) < 8:
        raise RuntimeError(f"Não existem imagens normais AeBAD-S suficientes em {replay_root}.")
    replay_rng = random.Random(42)
    replay_rng.shuffle(replay_candidates)
    replay_paths = replay_candidates[: min(REPLAY_NORMAL_IMAGES, len(replay_candidates))]
    training_paths = list(human_train_cases.values()) + replay_paths

    active_weights = MODELS_DIR / "blade_autoencoder.pt"
    if not active_weights.exists():
        raise RuntimeError("Pesos ativos do autoencoder ausentes.")
    checkpoint = torch.load(active_weights, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = BladeAutoencoder()
    model.load_state_dict(state_dict)

    dataset = NormalImageDataset(training_paths)
    loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.L1Loss()
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_state = copy.deepcopy(model.state_dict())
            best_loss = mean_loss
            best_epoch = epoch
    if best_state is None:
        raise RuntimeError("O fine-tuning do autoencoder não produziu pesos válidos.")
    model.load_state_dict(best_state)
    model.eval()

    calibration_scores = _score_paths(model, training_paths)
    calibration = {
        "top_error_p50": float(np.percentile(calibration_scores, 50)),
        "top_error_p95": float(np.percentile(calibration_scores, 95)),
        "top_error_p99": float(np.percentile(calibration_scores, 99)),
    }
    human_holdout_scores = _score_paths(model, list(human_holdout_cases.values()))
    public_paths = _normal_paths(DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "test" / "good" / "view")
    public_paths = public_paths[:PUBLIC_NORMAL_REGRESSION_CAP]
    public_scores = _score_paths(model, public_paths)

    baseline_metadata = _load_active_metadata()
    metrics: dict[str, Any] = {
        "approved_cohort_cases": len(candidate_ids),
        "usable_human_normal_cases": len(case_paths),
        "human_training_cases": len(human_train_cases),
        "human_holdout_cases": len(human_holdout_cases),
        "replay_normal_images": len(replay_paths),
        "total_training_images": len(training_paths),
        "best_epoch": best_epoch,
        "training_l1": round(best_loss, 6),
        "calibration": calibration,
        "human_normal_holdout": _summary(human_holdout_scores),
        "public_normal_view_regression": _summary(public_scores),
    }
    metadata = {
        "version_name": version_name,
        "model_type": "autoencoder_normal_adaptation_candidate",
        "base_model": baseline_metadata.get("version_name"),
        "input_size": MODEL_INPUT_SIZE,
        "training_target": "primary normality autoencoder only",
        "training": {
            "human_approved_normal_cases": len(case_paths),
            "human_training_cases": len(human_train_cases),
            "human_holdout_cases": len(human_holdout_cases),
            "public_normal_replay": {"dataset": "AeBAD-S/train/good", "images": len(replay_paths)},
            "epochs": FINE_TUNE_EPOCHS,
            "loss": "L1 reconstruction loss",
        },
        "calibration": calibration,
        "validation": {
            "human_holdout": "20% das peças humanas aprovadas, separado antes do treino",
            "public_normal_regression": "AeBAD-S/test/good/view; controlo de regressão, não conjunto de certificação",
            "human_normal_holdout": _summary(human_holdout_scores),
            "public_normal_view_regression": _summary(public_scores),
        },
        "metrics": metrics,
        "promotion": {
            "automatic": False,
            "required_action": "revisão humana das métricas e teste independente de anomalias antes de promoção",
        },
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{version_name}.pt"
    metadata_path = MODELS_DIR / f"{version_name}.json"
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, checkpoint_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with connect() as connection:
        _mark_version(
            connection,
            version_name,
            "validation_required",
            metrics,
            "Candidato do autoencoder primário ajustado apenas com imagens aprovadas como normais e replay AeBAD-S. Não foi promovido automaticamente.",
            {
                "usable_case_ids": sorted(case_paths),
                "human_training_case_ids": sorted(human_train_cases),
                "human_holdout_case_ids": sorted(human_holdout_cases),
                "checkpoint": checkpoint_path.name,
                "metadata": metadata_path.name,
            },
        )
        for inspection_id in case_paths:
            append_audit_event(
                connection,
                inspection_id,
                "local_training",
                "active_learning_primary_normal_candidate_completed",
                {"version_name": version_name, "checkpoint": checkpoint_path.name},
            )
        connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria um candidato do autoencoder a partir de aprovações humanas.")
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
                "O candidato do autoencoder falhou antes de poder ser validado; o modelo em produção não foi alterado.",
            )
        raise


if __name__ == "__main__":
    main()
