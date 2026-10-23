from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# The public delivery keeps the dashboard and training code in separate
# folders.  Resolve the repository root rather than assuming this script lives
# directly under it, then expose the dashboard package to Python imports.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATA_DIR, DATASETS_DIR, WORKSPACE_ROOT  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDS = (
    "source_dataset",
    "source_split",
    "sample_id",
    "image_path",
    "mask_path",
    "annotation_path",
    "paired_image_path",
    "original_label",
    "is_normal",
    "assignment",
    "role",
    "is_aircraft",
    "is_synthetic",
    "selection_note",
)


def _images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _relative(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _pick_evenly(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or len(paths) <= count:
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    positions = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[position] for position in sorted(set(positions))]


def _row(**values: Any) -> dict[str, str]:
    return {field: str(values.get(field, "")) for field in FIELDS}


def _aebad_static_rows() -> list[dict[str, str]]:
    root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S" / "train" / "good"
    return [
        _row(
            source_dataset="AeBAD-S",
            source_split="train/good",
            sample_id=path.stem,
            image_path=_relative(path),
            original_label="good",
            is_normal=1,
            assignment="primary_static_normality",
            role="train_or_replay_only",
            is_aircraft=1,
            is_synthetic=0,
            selection_note="Normal real de pá; não misturar defeitos neste modelo.",
        )
        for path in _images(root)
    ]


def _aebad_video_rows() -> list[dict[str, str]]:
    root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_V" / "train" / "good"
    return [
        _row(
            source_dataset="AeBAD-V",
            source_split="train/good",
            sample_id=path.relative_to(root).with_suffix("").as_posix(),
            image_path=_relative(path),
            original_label="good",
            is_normal=1,
            assignment="primary_video_normality",
            role="separate_video_profile_only",
            is_aircraft=1,
            is_synthetic=0,
            selection_note="Fotograma normal; manter cada vídeo inteiro no mesmo split e não misturar com o perfil estático.",
        )
        for path in _images(root)
    ]


def _bladesynth_rows(per_class: int) -> list[dict[str, str]]:
    images_root = DATASETS_DIR / "bladesynth" / "Dataset_bladesynth" / "Images"
    masks_root = DATASETS_DIR / "bladesynth" / "Dataset_bladesynth" / "Segmentation_masks"
    rows: list[dict[str, str]] = []
    for label in ("Normal", "Dent", "Nick", "Scratch", "Corrosion"):
        selected = _pick_evenly(_images(images_root / label), per_class)
        validation_cut = max(1, round(len(selected) * 0.15)) if len(selected) > 1 else 0
        for index, path in enumerate(selected):
            mask = masks_root / label / path.name if label != "Normal" else None
            if label != "Normal" and (mask is None or not mask.exists()):
                continue
            rows.append(
                _row(
                    source_dataset="BladeSynth",
                    source_split="synthetic",
                    sample_id=f"{label}/{path.stem}",
                    image_path=_relative(path),
                    mask_path=_relative(mask),
                    original_label=label,
                    is_normal=int(label == "Normal"),
                    assignment="resnet_spatial_pretrain",
                    role="pretrain_validation" if index < validation_cut else "pretrain_train",
                    is_aircraft=1,
                    is_synthetic=1,
                    selection_note="Pré-treino binário normal/defeito com máscara; a cabeça final AeBAD é afinada separadamente.",
                )
            )
    return rows


def _dlr_rows() -> list[dict[str, str]]:
    root = DATASETS_DIR / "dlr_aircraft_dent"
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        for path in _images(root / split / "images"):
            mask = root / split / "masks" / path.name.replace("-image", "-masks")
            annotation = root / split / "labels" / f"{path.stem}.txt"
            has_annotation = annotation.exists() and annotation.read_text(encoding="utf-8").strip()
            rows.append(
                _row(
                    source_dataset="DLR Aircraft Dent",
                    source_split=split,
                    sample_id=path.stem,
                    image_path=_relative(path),
                    mask_path=_relative(mask),
                    annotation_path=_relative(annotation),
                    original_label="dent" if has_annotation else "normal",
                    is_normal=int(not has_annotation),
                    assignment="binary_aircraft_dent_localisation",
                    role=f"official_{split}",
                    is_aircraft=1,
                    is_synthetic=0,
                    selection_note="Manter como cabeça binária/localização de amolgadela; não mapear para as classes AeBAD.",
                )
            )
    return rows


def _agdd_rows() -> list[dict[str, str]]:
    root = DATASETS_DIR / "agdd" / "data"
    rows: list[dict[str, str]] = []
    for split in ("train", "val"):
        for path in _images(root / "image" / split):
            paired = root / "images" / split / path.name
            annotation = root / "labels_rect" / split / f"{path.stem}.txt"
            rows.append(
                _row(
                    source_dataset="AGDD",
                    source_split=split,
                    sample_id=path.stem,
                    image_path=_relative(path),
                    paired_image_path=_relative(paired),
                    annotation_path=_relative(annotation),
                    original_label="agdd_annotated_defect",
                    is_normal="",
                    assignment="aircraft_canopy_obb_localisation",
                    role=f"official_{split}",
                    is_aircraft=1,
                    is_synthetic=0,
                    selection_note="Manter os pares frontal/traseiro no mesmo split e preservar OBB/caixa nativa; não mapear para classes AeBAD.",
                )
            )
    return rows


def _auxiliary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    mvtec = DATASETS_DIR / "mvtec_ad_metal_nut" / "metal_nut"
    for split in ("train", "test"):
        for label_dir in sorted(path for path in (mvtec / split).iterdir() if path.is_dir()) if (mvtec / split).exists() else []:
            for path in _images(label_dir):
                label = label_dir.name
                rows.append(
                    _row(
                        source_dataset="MVTec AD metal_nut",
                        source_split=f"{split}/{label}",
                        sample_id=f"{split}/{label}/{path.stem}",
                        image_path=_relative(path),
                        original_label=label,
                        is_normal=int(label == "good"),
                        assignment="industrial_metal_robustness",
                        role="auxiliary_train" if split == "train" else "external_eval_only",
                        is_aircraft=0,
                        is_synthetic=0,
                        selection_note="Robustez de textura metálica; não usar como métrica aeronáutica principal.",
                    )
                )

    kolektor = DATASETS_DIR / "kolektor_sdd2" / "extracted"
    for split in ("train", "test"):
        for path in _images(kolektor / split):
            if path.stem.endswith("_GT") or "copy" in path.name.lower():
                continue
            mask = path.with_name(f"{path.stem}_GT{path.suffix}")
            rows.append(
                _row(
                    source_dataset="KolektorSDD2",
                    source_split=split,
                    sample_id=f"{split}/{path.stem}",
                    image_path=_relative(path),
                    mask_path=_relative(mask),
                    original_label="binary_defect_or_normal",
                    is_normal="",
                    assignment="industrial_binary_localisation",
                    role="auxiliary_only",
                    is_aircraft=0,
                    is_synthetic=0,
                    selection_note="Usar apenas para pré-treino/robustez binária; o duplicado 10301 copy é excluído.",
                )
            )

    neu = DATASETS_DIR / "neu_det" / "extracted" / "NEU-DET"
    for path in _images(neu / "IMAGES"):
        annotation = neu / "ANNOTATIONS" / f"{path.stem}.xml"
        rows.append(
            _row(
                source_dataset="NEU-DET",
                source_split="official_unsplit",
                sample_id=path.stem,
                image_path=_relative(path),
                annotation_path=_relative(annotation),
                original_label=path.stem.rsplit("_", 1)[0],
                is_normal=0,
                assignment="steel_defect_spatial_pretrain",
                role="auxiliary_only",
                is_aircraft=0,
                is_synthetic=0,
                selection_note="Pré-treino de morfologias localizadas; não possui exemplos normais e não decide bom/defeito sozinho.",
            )
        )
    return rows


def _write_manifest(destination: Path, rows: Iterable[dict[str, str]]) -> int:
    materialized = list(rows)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def _manifest_path_exists(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    return (path if path.is_absolute() else WORKSPACE_ROOT / path).exists()


def _integrity(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    materialized = list(rows)
    images_present = sum(1 for row in materialized if _manifest_path_exists(row["image_path"]))
    if images_present != len(materialized):
        raise RuntimeError("Um manifesto contém uma imagem que deixou de existir no disco local.")
    report = {"images_verified": images_present}
    for field in ("mask_path", "annotation_path", "paired_image_path"):
        provided = [row[field] for row in materialized if row[field]]
        report[f"{field}_provided"] = len(provided)
        report[f"{field}_verified"] = sum(1 for value in provided if _manifest_path_exists(value))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria manifestos auditáveis sem copiar ou fundir datasets incompatíveis.")
    parser.add_argument("--bladesynth-per-class", type=int, default=1000)
    args = parser.parse_args()
    if args.bladesynth_per_class < 1:
        raise RuntimeError("--bladesynth-per-class tem de ser pelo menos 1.")

    destination = DATA_DIR / "manifests"
    destination.mkdir(parents=True, exist_ok=True)
    manifests = {
        "primary_static_normality_aebad_s.csv": _aebad_static_rows(),
        "primary_video_normality_aebad_v.csv": _aebad_video_rows(),
        "resnet_spatial_pretrain_bladesynth.csv": _bladesynth_rows(args.bladesynth_per_class),
        "aircraft_binary_localisation_dlr.csv": _dlr_rows(),
        "aircraft_canopy_localisation_agdd.csv": _agdd_rows(),
        "auxiliary_robustness_external.csv": _auxiliary_rows(),
    }
    counts = {name: _write_manifest(destination / name, rows) for name, rows in manifests.items()}
    summary = {
        "generated_at": "local build",
        "policy": {
            "primary_static": "AeBAD-S normal images only; user-approved target captures join later through active learning.",
            "primary_video": "AeBAD-V is isolated for a future video profile and is never mixed into static calibration.",
            "resnet": "BladeSynth is generic spatial pretraining only; AeBAD-S supplies the final four-class taxonomy.",
            "external": "DLR, AGDD, MVTec, Kolektor and NEU retain native labels and separate tasks.",
        },
        "manifest_counts": counts,
        "by_assignment": {
            name: dict(Counter(row["assignment"] for row in rows))
            for name, rows in manifests.items()
        },
        "integrity": {name: _integrity(rows) for name, rows in manifests.items()},
    }
    (destination / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(destination), **counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
