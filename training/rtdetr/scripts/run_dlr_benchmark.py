"""Train and compare two *candidate* DLR Aircraft Dent models.

The benchmark intentionally keeps the final DLR test images outside both model
trainings. It creates a CPU-feasible RT-DETR dent detector, a high-resolution
normality autoencoder, and a reproducible result bundle. Neither candidate
replaces a dashboard model automatically.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.models.autoencoder import BladeAutoencoder  # noqa: E402


DLR_ROOT = WORKSPACE_ROOT / "datasets" / "dlr_aircraft_dent"
AEBAD_NORMAL_ROOT = WORKSPACE_ROOT / "datasets" / "aebad" / "AeBAD" / "AeBAD_S" / "train" / "good"
# Runtime outputs are intentionally kept outside the committed evidence bundle.
# `dashboard/data` is ignored by Git and is also the local runtime location used
# by the dashboard.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dashboard" / "data" / "runs"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEST_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class DlrRecord:
    sample_id: str
    official_split: str
    image_path: Path
    mask_path: Path
    has_dent: bool


@dataclass(frozen=True)
class PreparedProtocol:
    rtdetr_train: tuple[DlrRecord, ...]
    rtdetr_validation: tuple[DlrRecord, ...]
    test: tuple[DlrRecord, ...]
    autoencoder_dlr_train_normals: tuple[DlrRecord, ...]
    autoencoder_calibration_normals: tuple[DlrRecord, ...]
    autoencoder_calibration_dents: tuple[DlrRecord, ...]


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _mask_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "masks" / image_path.name.replace("-image", "-masks")


def _mask_boxes(mask_path: Path, *, min_area: int = 12) -> list[tuple[int, int, int, int]]:
    """Return connected mask components as x1, y1, x2, y2."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Máscara DLR inacessível: {mask_path}")
    binary = (mask > 0).astype(np.uint8)
    components, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for component_index in range(1, components):
        x, y, width, height, area = (int(item) for item in stats[component_index])
        if area >= min_area and width > 0 and height > 0:
            boxes.append((x, y, x + width, y + height))
    return boxes


def _read_dlr_records() -> list[DlrRecord]:
    if not DLR_ROOT.exists():
        raise RuntimeError(f"Dataset DLR não encontrado em {DLR_ROOT}.")
    records: list[DlrRecord] = []
    for official_split in ("train", "val", "test"):
        image_dir = DLR_ROOT / official_split / "images"
        if not image_dir.exists():
            raise RuntimeError(f"Diretório de imagens DLR ausente: {image_dir}")
        for image_path in _image_paths(image_dir):
            mask_path = _mask_for_image(image_path)
            if not mask_path.exists():
                raise RuntimeError(f"Máscara DLR correspondente ausente: {mask_path}")
            records.append(
                DlrRecord(
                    sample_id=f"{official_split}/{image_path.stem}",
                    official_split=official_split,
                    image_path=image_path,
                    mask_path=mask_path,
                    has_dent=bool(_mask_boxes(mask_path)),
                )
            )
    if not records:
        raise RuntimeError("Não foram encontradas imagens DLR.")
    return records


def _sample_stably(records: Sequence[DlrRecord], count: int, seed: int) -> tuple[DlrRecord, ...]:
    if count > len(records):
        raise ValueError(f"Não é possível escolher {count} elementos de apenas {len(records)}.")
    generator = random.Random(seed)
    selected = generator.sample(sorted(records, key=lambda item: item.sample_id), count)
    return tuple(sorted(selected, key=lambda item: item.sample_id))


def _prepare_protocol(
    records: list[DlrRecord],
    seed: int,
    test_fraction: float,
    autoencoder_dlr_fraction: float,
) -> PreparedProtocol:
    """Seal a stratified test subset from the official DLR test partition."""

    by_split: dict[str, list[DlrRecord]] = defaultdict(list)
    by_label: dict[bool, list[DlrRecord]] = defaultdict(list)
    for record in records:
        by_split[record.official_split].append(record)
        by_label[record.has_dent].append(record)

    final_test: list[DlrRecord] = []
    for has_dent in (False, True):
        desired = int(round(len(by_label[has_dent]) * test_fraction))
        eligible = [record for record in by_split["test"] if record.has_dent == has_dent]
        final_test.extend(_sample_stably(eligible, desired, seed + int(has_dent)))
    final_test_tuple = tuple(sorted(final_test, key=lambda item: item.sample_id))
    final_test_ids = {record.sample_id for record in final_test_tuple}

    train_pool_normals = [
        record
        for record in records
        if record.official_split in {"train", "val"} and not record.has_dent
    ]
    target_autoencoder_normals = int(round(len(by_label[False]) * autoencoder_dlr_fraction))
    autoencoder_train_normals = _sample_stably(train_pool_normals, target_autoencoder_normals, seed + 100)
    autoencoder_train_ids = {record.sample_id for record in autoencoder_train_normals}
    calibration_normals = tuple(
        record
        for record in sorted(train_pool_normals, key=lambda item: item.sample_id)
        if record.sample_id not in autoencoder_train_ids
    )
    calibration_dents = tuple(
        record
        for record in sorted(by_split["val"], key=lambda item: item.sample_id)
        if record.has_dent
    )
    rtdetr_train = tuple(sorted(by_split["train"], key=lambda item: item.sample_id))
    rtdetr_validation = tuple(sorted(by_split["val"], key=lambda item: item.sample_id))
    assert not (final_test_ids & {record.sample_id for record in rtdetr_train})
    assert not (final_test_ids & {record.sample_id for record in rtdetr_validation})
    assert not (final_test_ids & autoencoder_train_ids)
    if not calibration_normals:
        raise RuntimeError("Não restaram normais DLR para calibração do autoencoder.")
    return PreparedProtocol(
        rtdetr_train=rtdetr_train,
        rtdetr_validation=rtdetr_validation,
        test=final_test_tuple,
        autoencoder_dlr_train_normals=autoencoder_train_normals,
        autoencoder_calibration_normals=calibration_normals,
        autoencoder_calibration_dents=calibration_dents,
    )


def _roles_for_protocol(protocol: PreparedProtocol) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = defaultdict(list)
    for name, records in (
        ("rtdetr_train", protocol.rtdetr_train),
        ("rtdetr_validation", protocol.rtdetr_validation),
        ("final_test", protocol.test),
        ("autoencoder_train_normal", protocol.autoencoder_dlr_train_normals),
        ("autoencoder_calibration_normal", protocol.autoencoder_calibration_normals),
        ("autoencoder_calibration_dent", protocol.autoencoder_calibration_dents),
    ):
        for record in records:
            roles[record.sample_id].append(name)
    return roles


def _write_protocol_manifest(protocol: PreparedProtocol, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    roles = _roles_for_protocol(protocol)
    all_records: dict[str, DlrRecord] = {}
    for records in (
        protocol.rtdetr_train,
        protocol.rtdetr_validation,
        protocol.test,
        protocol.autoencoder_dlr_train_normals,
        protocol.autoencoder_calibration_normals,
        protocol.autoencoder_calibration_dents,
    ):
        for record in records:
            all_records[record.sample_id] = record
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "official_split", "ground_truth", "image_path", "mask_path", "roles"),
        )
        writer.writeheader()
        for sample_id, record in sorted(all_records.items()):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "official_split": record.official_split,
                    "ground_truth": "dent" if record.has_dent else "normal",
                    "image_path": record.image_path.as_posix(),
                    "mask_path": record.mask_path.as_posix(),
                    "roles": ";".join(roles[sample_id]),
                }
            )


def _load_protocol_manifest(path: Path) -> PreparedProtocol:
    """Resume a prepared run without redrawing its sealed test subset."""

    if not path.exists():
        raise RuntimeError(f"Manifesto de protocolo inexistente: {path}")
    by_role: dict[str, list[DlrRecord]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = DlrRecord(
                sample_id=str(row["sample_id"]),
                official_split=str(row["official_split"]),
                image_path=Path(str(row["image_path"])),
                mask_path=Path(str(row["mask_path"])),
                has_dent=str(row["ground_truth"]) == "dent",
            )
            for role in str(row["roles"]).split(";"):
                by_role[role].append(record)
    required = (
        "rtdetr_train",
        "rtdetr_validation",
        "final_test",
        "autoencoder_train_normal",
        "autoencoder_calibration_normal",
        "autoencoder_calibration_dent",
    )
    missing = [role for role in required if not by_role[role]]
    if missing:
        raise RuntimeError(f"Manifesto de protocolo incompleto; faltam papéis: {', '.join(missing)}")
    return PreparedProtocol(
        rtdetr_train=tuple(sorted(by_role["rtdetr_train"], key=lambda item: item.sample_id)),
        rtdetr_validation=tuple(sorted(by_role["rtdetr_validation"], key=lambda item: item.sample_id)),
        test=tuple(sorted(by_role["final_test"], key=lambda item: item.sample_id)),
        autoencoder_dlr_train_normals=tuple(
            sorted(by_role["autoencoder_train_normal"], key=lambda item: item.sample_id)
        ),
        autoencoder_calibration_normals=tuple(
            sorted(by_role["autoencoder_calibration_normal"], key=lambda item: item.sample_id)
        ),
        autoencoder_calibration_dents=tuple(
            sorted(by_role["autoencoder_calibration_dent"], key=lambda item: item.sample_id)
        ),
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _yolo_line(box: tuple[int, int, int, int], image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    centre_x = (x1 + x2) / 2 / image_width
    centre_y = (y1 + y2) / 2 / image_height
    return f"0 {centre_x:.7f} {centre_y:.7f} {width / image_width:.7f} {height / image_height:.7f}"


def _materialize_rtdetr_dataset(protocol: PreparedProtocol, destination: Path) -> Path:
    """Create a DLR-only YOLO-format view without changing original data."""

    split_records = {
        "train": protocol.rtdetr_train,
        "val": protocol.rtdetr_validation,
        "test": protocol.test,
    }
    for split_name, records in split_records.items():
        for record in records:
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Imagem DLR inacessível: {record.image_path}")
            height, width = image.shape[:2]
            filename = f"{record.official_split}__{record.image_path.name}"
            image_target = destination / "images" / split_name / filename
            label_target = destination / "labels" / split_name / f"{Path(filename).stem}.txt"
            _link_or_copy(record.image_path, image_target)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text(
                "\n".join(_yolo_line(box, width, height) for box in _mask_boxes(record.mask_path)),
                encoding="utf-8",
            )
    config = {
        "path": destination.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "dent"},
    }
    yaml_path = destination / "dlr_aircraft_dent.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def _balanced_rtdetr_subset(
    records: Sequence[DlrRecord],
    maximum: int,
    seed: int,
) -> tuple[DlrRecord, ...]:
    """Select a reproducible, class-balanced CPU fine-tuning subset.

    DLR contains many fully normal frames.  A balanced subset lets the small
    RT-DETR see enough dents per CPU hour while all source images still come
    exclusively from the official DLR train/validation partitions.
    """

    ordered = tuple(sorted(records, key=lambda item: item.sample_id))
    if maximum <= 0 or maximum >= len(ordered):
        return ordered
    desired_dents = min(len([item for item in ordered if item.has_dent]), maximum // 2)
    desired_normals = min(len([item for item in ordered if not item.has_dent]), maximum - desired_dents)
    remaining = maximum - desired_dents - desired_normals
    if remaining:
        # Retain the original label proportions only after both classes have
        # received their initial balanced allocation.
        candidates = [item for item in ordered if item.has_dent][desired_dents:] + [
            item for item in ordered if not item.has_dent
        ][desired_normals:]
        extra = _sample_stably(candidates, min(remaining, len(candidates)), seed + 29)
    else:
        extra = ()
    dents = _sample_stably([item for item in ordered if item.has_dent], desired_dents, seed + 11)
    normals = _sample_stably([item for item in ordered if not item.has_dent], desired_normals, seed + 17)
    return tuple(sorted((*dents, *normals, *extra), key=lambda item: item.sample_id))


def _materialize_rtdetr_finetune_dataset(
    protocol: PreparedProtocol,
    destination: Path,
    *,
    train_maximum: int,
    validation_maximum: int,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    """Write a DLR-only fine-tuning view and its selection manifest.

    The final test is intentionally omitted from this YAML altogether.  It is
    evaluated later by the benchmark code, so the training framework cannot
    accidentally use it for selection or metrics.
    """

    selected_train = _balanced_rtdetr_subset(protocol.rtdetr_train, train_maximum, seed + 200)
    selected_validation = _balanced_rtdetr_subset(protocol.rtdetr_validation, validation_maximum, seed + 300)
    for split_name, records in (("train", selected_train), ("val", selected_validation)):
        for record in records:
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Imagem DLR inacessível: {record.image_path}")
            height, width = image.shape[:2]
            filename = f"{record.official_split}__{record.image_path.name}"
            _link_or_copy(record.image_path, destination / "images" / split_name / filename)
            label_target = destination / "labels" / split_name / f"{Path(filename).stem}.txt"
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text(
                "\n".join(_yolo_line(box, width, height) for box in _mask_boxes(record.mask_path)),
                encoding="utf-8",
            )
    config = {
        "path": destination.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "dent"},
    }
    yaml_path = destination / "dlr_aircraft_dent_finetune.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest_path = destination.parent / "rtdetr_finetune_selection.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "sample_id", "ground_truth", "image_path", "mask_path"))
        writer.writeheader()
        for role, records in (("rtdetr_finetune_train", selected_train), ("rtdetr_finetune_validation", selected_validation)):
            for record in records:
                writer.writerow(
                    {
                        "role": role,
                        "sample_id": record.sample_id,
                        "ground_truth": "dent" if record.has_dent else "normal",
                        "image_path": record.image_path.as_posix(),
                        "mask_path": record.mask_path.as_posix(),
                    }
                )
    summary = {
        "architecture": "YOLOv8n-RT-DETR hybrid detector",
        "full_dlr_train_pool_images": len(protocol.rtdetr_train),
        "full_dlr_validation_pool_images": len(protocol.rtdetr_validation),
        "fine_tune_train_images": len(selected_train),
        "fine_tune_train_dents": sum(record.has_dent for record in selected_train),
        "fine_tune_train_normals": sum(not record.has_dent for record in selected_train),
        "fine_tune_validation_images": len(selected_validation),
        "fine_tune_validation_dents": sum(record.has_dent for record in selected_validation),
        "fine_tune_validation_normals": sum(not record.has_dent for record in selected_validation),
        "selection_manifest": manifest_path.as_posix(),
        "test_included": False,
    }
    (destination.parent / "rtdetr_finetune_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return yaml_path, summary


def _aebad_normal_paths() -> list[Path]:
    if not AEBAD_NORMAL_ROOT.exists():
        raise RuntimeError(f"Normais AeBAD-S ausentes: {AEBAD_NORMAL_ROOT}")
    paths = sorted(
        path
        for path in AEBAD_NORMAL_ROOT.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError("Não foram encontradas imagens normais AeBAD-S.")
    return paths


def _letterbox_rgb(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Preserve aspect ratio and return a valid-pixel mask for loss/scoring."""

    height, width = image_bgr.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    valid = np.zeros((size, size), dtype=np.float32)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized
    valid[y : y + new_height, x : x + new_width] = 1.0
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), valid


def _restore_error_map(error: np.ndarray, original_shape: tuple[int, int], valid: np.ndarray) -> np.ndarray:
    height, width = original_shape
    locations = np.argwhere(valid > 0)
    if locations.size == 0:
        raise RuntimeError("Máscara de regiões válidas do autoencoder vazia.")
    y1, x1 = locations.min(axis=0)
    y2, x2 = locations.max(axis=0) + 1
    crop = error[int(y1) : int(y2), int(x1) : int(x2)]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)


class NormalImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A memory-mapped cache avoids re-decoding full-resolution PNGs per epoch."""

    def __init__(
        self,
        paths: Sequence[Path],
        input_size: int,
        augment: bool,
        cache_dir: Path,
        cache_name: str,
    ) -> None:
        self.paths = list(paths)
        self.input_size = input_size
        self.augment = augment
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.images, self.bounds = self._open_or_create_cache()

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.input_size).encode("utf-8"))
        for path in self.paths:
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _open_or_create_cache(self) -> tuple[np.ndarray, np.ndarray]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.cache_dir / f"{self.cache_name}_{self.input_size}.npy"
        bounds_path = self.cache_dir / f"{self.cache_name}_{self.input_size}_bounds.npy"
        metadata_path = self.cache_dir / f"{self.cache_name}_{self.input_size}.json"
        key = self._cache_key()
        if image_path.exists() and bounds_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("key") == key and metadata.get("count") == len(self.paths):
                    return np.load(image_path, mmap_mode="r"), np.load(bounds_path, mmap_mode="r")
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        images = np.lib.format.open_memmap(
            image_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(self.paths), self.input_size, self.input_size, 3),
        )
        bounds = np.lib.format.open_memmap(
            bounds_path,
            mode="w+",
            dtype=np.int16,
            shape=(len(self.paths), 4),
        )
        for index, path in enumerate(self.paths, start=1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Imagem normal inacessível: {path}")
            rgb, valid = _letterbox_rgb(image, self.input_size)
            locations = np.argwhere(valid > 0)
            y1, x1 = locations.min(axis=0)
            y2, x2 = locations.max(axis=0) + 1
            images[index - 1] = rgb
            bounds[index - 1] = (x1, y1, x2, y2)
            if index % 100 == 0 or index == len(self.paths):
                print(
                    json.dumps(
                        {
                            "component": "autoencoder_cache",
                            "cache": self.cache_name,
                            "prepared": index,
                            "total": len(self.paths),
                        }
                    ),
                    flush=True,
                )
        images.flush()
        bounds.flush()
        metadata_path.write_text(json.dumps({"key": key, "count": len(self.paths)}, indent=2), encoding="utf-8")
        return np.load(image_path, mmap_mode="r"), np.load(bounds_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(self.images[index], copy=True)
        x1, y1, x2, y2 = (int(value) for value in self.bounds[index])
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
        if self.augment:
            image = cv2.convertScaleAbs(
                image,
                alpha=random.uniform(0.92, 1.08),
                beta=random.uniform(-10.0, 10.0),
            )
        valid = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        valid[y1:y2, x1:x2] = 1.0
        return (
            torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.0),
            torch.from_numpy(valid.copy()).unsqueeze(0),
        )


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    expanded = valid.expand_as(target)
    return (torch.abs(prediction - target) * expanded).sum() / expanded.sum().clamp_min(1.0)


def _masked_gradient_l1(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_x = valid[:, :, :, 1:] * valid[:, :, :, :-1]
    valid_y = valid[:, :, 1:, :] * valid[:, :, :-1, :]
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = (torch.abs(pred_x - target_x) * valid_x.expand_as(pred_x)).sum() / valid_x.expand_as(pred_x).sum().clamp_min(1.0)
    loss_y = (torch.abs(pred_y - target_y) * valid_y.expand_as(pred_y)).sum() / valid_y.expand_as(pred_y).sum().clamp_min(1.0)
    return (loss_x + loss_y) / 2.0


def _autoencoder_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return 0.88 * _masked_l1(prediction, target, valid) + 0.12 * _masked_gradient_l1(prediction, target, valid)


def _autoencoder_tensor_and_error(
    image_path: Path,
    model: BladeAutoencoder,
    input_size: int,
) -> tuple[float, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Imagem inacessível: {image_path}")
    rgb, valid = _letterbox_rgb(image, input_size)
    source = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.inference_mode():
        reconstruction = model(source)
    error = torch.mean(torch.abs(source - reconstruction), dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    error = cv2.GaussianBlur(error, (0, 0), sigmaX=1.1)
    score = float(np.percentile(error[valid > 0], 99.2))
    return score, _restore_error_map(error, image.shape[:2], valid)


def _image_classification_metrics(labels: Sequence[int], predictions: Sequence[bool]) -> dict[str, float]:
    tp = sum(label == 1 and bool(prediction) for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and not bool(prediction) for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and bool(prediction) for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and not bool(prediction) for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "accuracy": float((tp + tn) / max(1, len(labels))),
        "precision_dent": float(precision),
        "recall_dent": float(recall),
        "f1_dent": float(f1),
        "specificity_normal": float(specificity),
    }


def _choose_image_threshold(normal_scores: Sequence[float], dent_scores: Sequence[float]) -> tuple[float, dict[str, float]]:
    """Select the image threshold on development validation only."""

    scores = list(normal_scores) + list(dent_scores)
    labels = [0] * len(normal_scores) + [1] * len(dent_scores)
    if not scores or not any(labels):
        raise RuntimeError("Calibração do autoencoder requer normais e amolgadelas DLR.")
    candidates = sorted(set(scores))
    if len(candidates) > 240:
        candidates = [float(value) for value in np.linspace(min(scores), max(scores), 240)]
    best: tuple[float, float, float] | None = None
    selected = candidates[0]
    best_metrics: dict[str, float] = {}
    for threshold in candidates:
        metrics = _image_classification_metrics(labels, [score >= threshold for score in scores])
        candidate = (metrics["f1_dent"], metrics["specificity_normal"], -threshold)
        if best is None or candidate > best:
            best = candidate
            selected = float(threshold)
            best_metrics = metrics
    return selected, {key: float(value) for key, value in best_metrics.items()}


def _train_autoencoder(
    protocol: PreparedProtocol,
    run_dir: Path,
    *,
    input_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    """Train a high-resolution candidate without overwriting dashboard weights."""

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    aebad_paths = _aebad_normal_paths()
    dlr_paths = [record.image_path for record in protocol.autoencoder_dlr_train_normals]
    calibration_paths = [record.image_path for record in protocol.autoencoder_calibration_normals]
    auto_dir = run_dir / "autoencoder"
    auto_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = NormalImageDataset(
        aebad_paths + dlr_paths,
        input_size,
        augment=True,
        cache_dir=auto_dir / "preprocessed_cache",
        cache_name="train",
    )
    validation_dataset = NormalImageDataset(
        calibration_paths,
        input_size,
        augment=False,
        cache_dir=auto_dir / "preprocessed_cache",
        cache_name="validation",
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = BladeAutoencoder()
    optimizer = AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    no_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for source, valid in train_loader:
            noisy = (source + torch.randn_like(source) * 0.012).clamp(0.0, 1.0)
            reconstruction = model(noisy)
            loss = _autoencoder_loss(reconstruction, source, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()
        model.eval()
        validation_losses: list[float] = []
        with torch.inference_mode():
            for source, valid in validation_loader:
                validation_losses.append(float(_autoencoder_loss(model(source), source, valid).item()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss})
        print(
            json.dumps(
                {
                    "component": "autoencoder",
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "validation_loss": round(validation_loss, 6),
                }
            ),
            flush=True,
        )
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 4:
                print("Autoencoder: early stopping por validação normal.", flush=True)
                break
    if best_state is None:
        raise RuntimeError("O treino do autoencoder não produziu checkpoint válido.")
    model.load_state_dict(best_state)
    model.eval()
    weights_path = auto_dir / "dlr_mixed_autoencoder.pt"
    torch.save({"state_dict": model.state_dict()}, weights_path)

    normal_scores: list[float] = []
    pixel_samples: list[np.ndarray] = []
    for path in calibration_paths:
        score, error = _autoencoder_tensor_and_error(path, model, input_size)
        normal_scores.append(score)
        pixel_samples.append(error.ravel()[::16])
    dent_scores = [
        _autoencoder_tensor_and_error(record.image_path, model, input_size)[0]
        for record in protocol.autoencoder_calibration_dents
    ]
    image_threshold, threshold_validation = _choose_image_threshold(normal_scores, dent_scores)
    pixel_threshold = float(np.percentile(np.concatenate(pixel_samples), 99.7))
    metadata = {
        "version_name": f"dlr_mixed_autoencoder_{_now_tag()}",
        "model_type": "candidate_normality_autoencoder",
        "status": "evaluation_only_not_dashboard_active",
        "architecture": "BladeAutoencoder convolutional denoising reconstruction model",
        "input_size": input_size,
        "loss": "0.88 masked L1 + 0.12 masked gradient L1",
        "epochs_completed": len(history),
        "best_validation_loss": best_validation,
        "training": {
            "aebad_s_train_good_images": len(aebad_paths),
            "dlr_normal_images": len(dlr_paths),
            "dlr_normal_fraction_of_full_dataset": 0.70,
            "total_normal_images": len(aebad_paths) + len(dlr_paths),
            "excluded_final_dlr_test_images": len(protocol.test),
        },
        "calibration": {
            "normal_dlr_images": len(normal_scores),
            "dent_dlr_images_for_validation": len(dent_scores),
            "image_threshold_selected_on_validation": image_threshold,
            "pixel_threshold_normal_p99_7": pixel_threshold,
            "validation_metrics": threshold_validation,
        },
        "history": history,
    }
    (auto_dir / "dlr_mixed_autoencoder.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return weights_path, metadata


def _load_autoencoder_candidate(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Reuse a completed candidate when retrying only the RT-DETR phase."""

    weights_path = run_dir / "autoencoder" / "dlr_mixed_autoencoder.pt"
    metadata_path = run_dir / "autoencoder" / "dlr_mixed_autoencoder.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise RuntimeError("Não existe candidato de autoencoder completo para reutilizar.")
    return weights_path, json.loads(metadata_path.read_text(encoding="utf-8"))


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(intersection / max(1e-9, first_area + second_area - intersection))


def _ground_truth_boxes(record: DlrRecord) -> list[tuple[int, int, int, int]]:
    return _mask_boxes(record.mask_path)


def _residual_boxes(
    error_map: np.ndarray,
    pixel_threshold: float,
    *,
    min_area_fraction: float = 0.00012,
) -> list[dict[str, float]]:
    binary = (error_map >= pixel_threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    components, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum_area = max(10, int(error_map.shape[0] * error_map.shape[1] * min_area_fraction))
    boxes: list[dict[str, float]] = []
    for index in range(1, components):
        x, y, width, height, area = (int(item) for item in stats[index])
        if area < minimum_area or width < 3 or height < 3:
            continue
        component = error_map[y : y + height, x : x + width]
        boxes.append(
            {
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + width),
                "y2": float(y + height),
                "confidence": float(np.percentile(component, 99.0)),
            }
        )
    return sorted(boxes, key=lambda item: item["confidence"], reverse=True)[:30]


def _time_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0, "p50_seconds": 0.0, "p95_seconds": 0.0}
    return {
        "mean_seconds": float(np.mean(values)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
    }


def _run_autoencoder_test(
    test_records: Sequence[DlrRecord],
    weights_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = BladeAutoencoder()
    model.load_state_dict(state_dict)
    model.eval()
    input_size = int(metadata["input_size"])
    threshold = float(metadata["calibration"]["image_threshold_selected_on_validation"])
    pixel_threshold = float(metadata["calibration"]["pixel_threshold_normal_p99_7"])
    _autoencoder_tensor_and_error(test_records[0].image_path, model, input_size)
    predictions: dict[str, dict[str, Any]] = {}
    times: list[float] = []
    for position, record in enumerate(test_records, start=1):
        started = time.perf_counter()
        score, error_map = _autoencoder_tensor_and_error(record.image_path, model, input_size)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        predictions[record.sample_id] = {
            "score": score,
            "image_prediction_dent": bool(score >= threshold),
            "boxes": _residual_boxes(error_map, pixel_threshold),
            "seconds": elapsed,
        }
        if position % 25 == 0 or position == len(test_records):
            print(f"Autoencoder: inferência {position}/{len(test_records)}", flush=True)
    return {
        "predictions": predictions,
        "timing": _time_statistics(times),
        "image_threshold": threshold,
        "pixel_threshold": pixel_threshold,
    }


def _train_rtdetr(
    data_yaml: Path,
    run_dir: Path,
    *,
    model_source: str,
    initialization_weights: str | None,
    epochs: int,
    image_size: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> Path:
    try:
        from ultralytics import RTDETR
    except ImportError as error:
        raise RuntimeError("Ultralytics não está instalado; instale-o antes de iniciar RT-DETR.") from error
    model = RTDETR(model_source)
    if initialization_weights:
        # This transfers a public generic visual backbone only.  Every
        # supervised update in this experiment is still DLR Aircraft Dent.
        model.load(initialization_weights)
    results_dir = run_dir / "rtdetr"
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device="cpu",
        workers=0,
        optimizer="AdamW",
        lr0=2e-4,
        lrf=0.02,
        cos_lr=True,
        warmup_epochs=1.0,
        patience=patience,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.12,
        hsv_v=0.12,
        erasing=0.0,
        translate=0.05,
        cache=False,
        amp=False,
        deterministic=True,
        seed=seed,
        plots=True,
        project=str(results_dir),
        name="finetune_n_final",
        exist_ok=True,
        verbose=True,
    )
    weights_dir = results_dir / "finetune_n_final" / "weights"
    if (weights_dir / "best.pt").exists():
        return weights_dir / "best.pt"
    if (weights_dir / "last.pt").exists():
        return weights_dir / "last.pt"
    raise RuntimeError(f"O treino RT-DETR não gerou pesos em {weights_dir}.")


def _run_rtdetr_test(
    test_records: Sequence[DlrRecord],
    weights_path: Path,
    *,
    image_size: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    from ultralytics import RTDETR

    model = RTDETR(str(weights_path))
    model.predict(str(test_records[0].image_path), imgsz=image_size, conf=0.001, iou=0.7, device="cpu", verbose=False)
    predictions: dict[str, dict[str, Any]] = {}
    times: list[float] = []
    for position, record in enumerate(test_records, start=1):
        started = time.perf_counter()
        result = model.predict(
            str(record.image_path),
            imgsz=image_size,
            conf=0.001,
            iou=0.7,
            max_det=100,
            device="cpu",
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        boxes: list[dict[str, float]] = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, confidence, class_index in zip(xyxy, confidences, classes):
                if int(class_index) == 0:
                    boxes.append(
                        {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "confidence": float(confidence),
                        }
                    )
        boxes.sort(key=lambda item: item["confidence"], reverse=True)
        predictions[record.sample_id] = {
            "score": max((box["confidence"] for box in boxes), default=0.0),
            "image_prediction_dent": bool(any(box["confidence"] >= confidence_threshold for box in boxes)),
            "boxes": boxes,
            "seconds": elapsed,
        }
        if position % 25 == 0 or position == len(test_records):
            print(f"RT-DETR: inferência {position}/{len(test_records)}", flush=True)
    return {
        "predictions": predictions,
        "timing": _time_statistics(times),
        "confidence_threshold": confidence_threshold,
    }


def _match_counts(
    test_records: Sequence[DlrRecord],
    predictions: dict[str, dict[str, Any]],
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, int]:
    true_positive = false_positive = false_negative = 0
    for record in test_records:
        ground_truth = _ground_truth_boxes(record)
        predicted = [
            (box["x1"], box["y1"], box["x2"], box["y2"], box["confidence"])
            for box in predictions[record.sample_id]["boxes"]
            if float(box["confidence"]) >= confidence_threshold
        ]
        matched: set[int] = set()
        for box in sorted(predicted, key=lambda item: item[4], reverse=True):
            best_index = -1
            best_iou = 0.0
            for index, target in enumerate(ground_truth):
                if index not in matched and (overlap := _iou(box[:4], target)) > best_iou:
                    best_iou, best_index = overlap, index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched.add(best_index)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(ground_truth) - len(matched)
    return {"tp": true_positive, "fp": false_positive, "fn": false_negative}


def _average_precision_detection(
    test_records: Sequence[DlrRecord],
    predictions: dict[str, dict[str, Any]],
    iou_threshold: float,
) -> float:
    ground_truths = {record.sample_id: _ground_truth_boxes(record) for record in test_records}
    total = sum(len(boxes) for boxes in ground_truths.values())
    if not total:
        return 0.0
    flattened: list[tuple[float, str, tuple[float, float, float, float]]] = []
    for record in test_records:
        for box in predictions[record.sample_id]["boxes"]:
            flattened.append(
                (
                    float(box["confidence"]),
                    record.sample_id,
                    (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                )
            )
    flattened.sort(reverse=True, key=lambda item: item[0])
    matched: dict[str, set[int]] = defaultdict(set)
    tp: list[int] = []
    fp: list[int] = []
    for _confidence, sample_id, box in flattened:
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(ground_truths[sample_id]):
            if index not in matched[sample_id] and (overlap := _iou(box, target)) > best_iou:
                best_iou, best_index = overlap, index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[sample_id].add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not tp:
        return 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recalls = cumulative_tp / total
    precisions = cumulative_tp / np.maximum(1, cumulative_tp + cumulative_fp)
    return float(
        np.mean(
            [
                np.max(precisions[recalls >= recall]) if np.any(recalls >= recall) else 0.0
                for recall in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def _evaluate_model(
    test_records: Sequence[DlrRecord],
    predictions: dict[str, dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    labels = [int(record.has_dent) for record in test_records]
    decisions = [bool(predictions[record.sample_id]["image_prediction_dent"]) for record in test_records]
    image_metrics = _image_classification_metrics(labels, decisions)
    counts = _match_counts(test_records, predictions, confidence_threshold, 0.50)
    detection_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    detection_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    ap_by_iou = {f"{threshold:.2f}": _average_precision_detection(test_records, predictions, threshold) for threshold in TEST_IOU_THRESHOLDS}
    return {
        "image_level": image_metrics,
        "detection_level_iou_0_50": {
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
            "precision": detection_precision,
            "recall": detection_recall,
        },
        "map_50": ap_by_iou["0.50"],
        "map_50_95": float(np.mean(list(ap_by_iou.values()))),
        "ap_by_iou": ap_by_iou,
    }


def _save_confusion_matrix(metrics: dict[str, Any], destination: Path, title: str) -> None:
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        dtype=int,
    )
    canvas = np.full((500, 760, 3), (248, 250, 252), dtype=np.uint8)
    cv2.putText(canvas, title, (38, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 45, 65), 2, cv2.LINE_AA)
    origin_x, origin_y, cell = 260, 125, 145
    cv2.putText(canvas, "Previsto normal", (origin_x - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 45, 65), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Previsto dent", (origin_x + cell + 8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 45, 65), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Real normal", (62, origin_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 45, 65), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Real dent", (78, origin_y + cell + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 45, 65), 1, cv2.LINE_AA)
    for row in range(2):
        for column in range(2):
            colour = (238 - row * 28, 190 + column * 24, 78 + row * 48)
            x1 = origin_x + column * cell
            y1 = origin_y + row * cell
            cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), colour, thickness=-1)
            cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), (45, 61, 80), thickness=2)
            value = str(matrix[row, column])
            text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 1.25, 2)[0]
            cv2.putText(
                canvas,
                value,
                (x1 + (cell - text_size[0]) // 2, y1 + (cell + text_size[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                (15, 28, 42),
                2,
                cv2.LINE_AA,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Não foi possível escrever matriz de confusão: {destination}")


def _draw_boxes(image: np.ndarray, boxes: Iterable[dict[str, float]], color: tuple[int, int, int], prefix: str) -> np.ndarray:
    drawn = image.copy()
    for box in boxes:
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        confidence = box.get("confidence")
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 4)
        label = prefix if confidence is None else f"{prefix} {float(confidence):.2f}"
        cv2.putText(drawn, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return drawn


def _header(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    banner = np.full((68, image.shape[1], 3), (18, 28, 43), dtype=np.uint8)
    cv2.putText(banner, text, (18, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    return np.vstack((banner, image))


def _scaled_panel(image: np.ndarray, width: int = 640) -> np.ndarray:
    height, original_width = image.shape[:2]
    return cv2.resize(image, (width, max(1, int(round(height * width / original_width)))), interpolation=cv2.INTER_AREA)


def _make_visual_examples(
    test_records: Sequence[DlrRecord],
    auto_predictions: dict[str, dict[str, Any]],
    rtdetr_predictions: dict[str, dict[str, Any]],
    destination: Path,
) -> dict[str, Any]:
    """Create 20 comparative panels: five mistakes when available, then correct cases."""

    candidates: list[tuple[DlrRecord, bool, str]] = []
    for record in test_records:
        truth = record.has_dent
        ae_correct = bool(auto_predictions[record.sample_id]["image_prediction_dent"]) == truth
        detector_correct = bool(rtdetr_predictions[record.sample_id]["image_prediction_dent"]) == truth
        if not ae_correct or not detector_correct:
            candidates.append((record, False, f"AE={'ok' if ae_correct else 'erro'}, RT-DETR={'ok' if detector_correct else 'erro'}"))
        else:
            candidates.append((record, True, "ambos corretos"))
    errors = [item for item in candidates if not item[1]]
    correct = [item for item in candidates if item[1]]
    selected = errors[:5] + correct[:15]
    if len(selected) < 20:
        selected.extend(errors[5 : 5 + (20 - len(selected))])
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for position, (record, is_correct, note) in enumerate(selected, start=1):
        original = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if original is None:
            continue
        truth_boxes = [
            {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
            for x1, y1, x2, y2 in _ground_truth_boxes(record)
        ]
        truth = "DENT" if record.has_dent else "NORMAL"
        original_panel = _header(
            _scaled_panel(_draw_boxes(original, truth_boxes, (61, 214, 110), "REAL")),
            f"Original | verdade: {truth}",
            (100, 235, 157),
        )
        ae = auto_predictions[record.sample_id]
        ae_label = "DENT" if ae["image_prediction_dent"] else "NORMAL"
        ae_panel = _header(
            _scaled_panel(_draw_boxes(original, ae["boxes"], (29, 157, 255), "AE")),
            f"Autoencoder | previsto: {ae_label} | score {ae['score']:.3f}",
            (116, 192, 255),
        )
        detector = rtdetr_predictions[record.sample_id]
        detector_label = "DENT" if detector["image_prediction_dent"] else "NORMAL"
        detector_panel = _header(
            _scaled_panel(_draw_boxes(original, detector["boxes"], (62, 73, 244), "RT-DETR")),
            f"RT-DETR | previsto: {detector_label} | score {detector['score']:.3f}",
            (115, 130, 255),
        )
        panel = np.hstack((original_panel, ae_panel, detector_panel))
        kind = "correto" if is_correct else "erro"
        filename = f"{position:02d}_{kind}_{record.image_path.stem}.jpg"
        output = destination / filename
        if not cv2.imwrite(str(output), panel, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Não foi possível escrever imagem de demonstração: {output}")
        manifest.append(
            {
                "file": filename,
                "sample_id": record.sample_id,
                "ground_truth": truth.lower(),
                "selection": kind,
                "note": note,
            }
        )
    manifest_path = destination / "seleccao_20_exemplos.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "created": len(manifest),
        "requested_errors": 5,
        "available_errors": len(errors),
        "manifest": manifest_path.as_posix(),
    }


def _box_area_fraction(record: DlrRecord) -> float | None:
    boxes = _ground_truth_boxes(record)
    if not boxes:
        return None
    image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return float(sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes) / (image.shape[0] * image.shape[1]))


def _brightness_stats(record: DlrRecord) -> tuple[float, float]:
    image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, 0.0
    return float(np.mean(image >= 248)), float(np.std(image))


def _error_observations(test_records: Sequence[DlrRecord], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    false_negatives = [
        record for record in test_records if record.has_dent and not predictions[record.sample_id]["image_prediction_dent"]
    ]
    false_positives = [
        record for record in test_records if not record.has_dent and predictions[record.sample_id]["image_prediction_dent"]
    ]
    true_positives = [
        record for record in test_records if record.has_dent and predictions[record.sample_id]["image_prediction_dent"]
    ]
    def mean_area(records: Sequence[DlrRecord]) -> float | None:
        values = [value for record in records if (value := _box_area_fraction(record)) is not None]
        return None if not values else float(np.mean(values))

    false_positive_brightness = [_brightness_stats(record) for record in false_positives]
    true_normal_brightness = [
        _brightness_stats(record)
        for record in test_records
        if not record.has_dent and not predictions[record.sample_id]["image_prediction_dent"]
    ]
    return {
        "false_negative_images": len(false_negatives),
        "false_positive_images": len(false_positives),
        "true_positive_images": len(true_positives),
        "mean_dent_area_fraction_false_negative": mean_area(false_negatives),
        "mean_dent_area_fraction_true_positive": mean_area(true_positives),
        "mean_specular_fraction_false_positive": None if not false_positive_brightness else float(np.mean([item[0] for item in false_positive_brightness])),
        "mean_specular_fraction_true_normal": None if not true_normal_brightness else float(np.mean([item[0] for item in true_normal_brightness])),
        "mean_contrast_false_positive": None if not false_positive_brightness else float(np.mean([item[1] for item in false_positive_brightness])),
        "mean_contrast_true_normal": None if not true_normal_brightness else float(np.mean([item[1] for item in true_normal_brightness])),
    }


def _format_number(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_error_study(
    destination: Path,
    *,
    results: dict[str, Any],
    auto_observations: dict[str, Any],
    rtdetr_observations: dict[str, Any],
) -> None:
    auto_metrics = results["autoencoder"]["metrics"]
    detector_metrics = results["rtdetr"]["metrics"]
    detector_training = results["rtdetr"]["training_data"]
    text = f"""# Estudo de erros — benchmark DLR Aircraft Dent

## Protocolo

- Teste final: {results["protocol"]["test_images"]} imagens DLR, seladas antes dos dois treinos.
- RT-DETR: seleção DLR balanceada de {detector_training["fine_tune_train_images"]} imagens para treino e {detector_training["fine_tune_validation_images"]} para validação; o teste selado não participou.
- Autoencoder: {results["protocol"]["aebad_normal_train_images"]} normais AeBAD-S + {results["protocol"]["dlr_normal_train_images"]} normais DLR; nenhum dent DLR no treino.
- O limiar do autoencoder foi escolhido no desenvolvimento, nunca no teste final.

## Resultado quantitativo

| Modelo | Accuracy imagem | Precisão dent | Recall dent | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Autoencoder residual | {_format_number(auto_metrics["image_level"]["accuracy"])} | {_format_number(auto_metrics["image_level"]["precision_dent"])} | {_format_number(auto_metrics["image_level"]["recall_dent"])} | {_format_number(auto_metrics["map_50"])} | {_format_number(auto_metrics["map_50_95"])} |
| RT-DETR | {_format_number(detector_metrics["image_level"]["accuracy"])} | {_format_number(detector_metrics["image_level"]["precision_dent"])} | {_format_number(detector_metrics["image_level"]["recall_dent"])} | {_format_number(detector_metrics["map_50"])} | {_format_number(detector_metrics["map_50_95"])} |

## Observações medidas

### Autoencoder

- Falsos negativos por imagem: {auto_observations["false_negative_images"]}.
- Falsos positivos por imagem: {auto_observations["false_positive_images"]}.
- Área média da dent nos falsos negativos: {_format_number(auto_observations["mean_dent_area_fraction_false_negative"], 6)} da imagem.
- Área média da dent nos verdadeiros positivos: {_format_number(auto_observations["mean_dent_area_fraction_true_positive"], 6)} da imagem.

### RT-DETR

- Falsos negativos por imagem: {rtdetr_observations["false_negative_images"]}.
- Falsos positivos por imagem: {rtdetr_observations["false_positive_images"]}.
- Área média da dent nos falsos negativos: {_format_number(rtdetr_observations["mean_dent_area_fraction_false_negative"], 6)} da imagem.
- Área média da dent nos verdadeiros positivos: {_format_number(rtdetr_observations["mean_dent_area_fraction_true_positive"], 6)} da imagem.

## Hipóteses a verificar nos painéis de demonstração

1. Dents pequenas ou pouco contrastadas podem perder-se na redução de escala ou não estar suficientemente representadas no treino.
2. Se falsos positivos forem mais brilhantes ou contrastados que normais corretamente rejeitados, reflexos especulares e textura superficial são hipóteses fortes.
3. O autoencoder aprende simultaneamente pás AeBAD-S e componentes DLR; a maior variedade de normalidade pode tornar o residual menos específico para dents DLR.
4. As caixas do autoencoder são componentes do mapa residual, não uma cabeça de deteção supervisionada. Um mAP inferior ao RT-DETR é esperado.
5. mAP@0.50:0.95 penaliza fortemente pequenas diferenças de fronteira em dents difusas.

Estas são hipóteses auditáveis; devem ser verificadas visualmente nos 20 painéis guardados na pasta visual_examples.
"""
    destination.write_text(text, encoding="utf-8")


def _protocol_summary(protocol: PreparedProtocol, aebad_count: int) -> dict[str, Any]:
    return {
        "rtdetr_train_images": len(protocol.rtdetr_train),
        "rtdetr_validation_images": len(protocol.rtdetr_validation),
        "test_images": len(protocol.test),
        "test_normals": sum(not record.has_dent for record in protocol.test),
        "test_dents": sum(record.has_dent for record in protocol.test),
        "aebad_normal_train_images": aebad_count,
        "dlr_normal_train_images": len(protocol.autoencoder_dlr_train_normals),
        "dlr_normal_calibration_images": len(protocol.autoencoder_calibration_normals),
        "dlr_dent_calibration_images": len(protocol.autoencoder_calibration_dents),
    }


def _run(args: argparse.Namespace) -> Path:
    if not 0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction deve estar entre 0 e 0.5.")
    if abs(args.autoencoder_dlr_fraction - 0.70) > 1e-9:
        raise ValueError("Este protocolo implementa exatamente 70% dos normais DLR no autoencoder.")
    if args.resume_run:
        run_dir = args.resume_run.resolve()
        protocol = _load_protocol_manifest(run_dir / "protocol_manifest.csv")
        base_data_yaml = run_dir / "rtdetr_dataset" / "dlr_aircraft_dent.yaml"
        if not base_data_yaml.exists():
            raise RuntimeError(f"Dataset RT-DETR preparado ausente: {base_data_yaml}")
    else:
        run_dir = (args.output_root / f"dlr_benchmark_{_now_tag()}").resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        protocol = _prepare_protocol(_read_dlr_records(), args.seed, args.test_fraction, args.autoencoder_dlr_fraction)
        _write_protocol_manifest(protocol, run_dir / "protocol_manifest.csv")
        base_data_yaml = _materialize_rtdetr_dataset(protocol, run_dir / "rtdetr_dataset")
    aebad_paths = _aebad_normal_paths()
    summary = _protocol_summary(protocol, len(aebad_paths))
    if not args.resume_run:
        (run_dir / "protocol.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "seed": args.seed,
                    "test_fraction_of_all_dlr": args.test_fraction,
                    "autoencoder_dlr_normal_fraction": args.autoencoder_dlr_fraction,
                    "summary": summary,
                    "rules": {
                        "test": "Imagens selecionadas de DLR/test; nenhuma entrou em treino, validação ou calibração.",
                        "rtdetr": "Somente DLR, com train para gradientes e val para seleção.",
                        "autoencoder": "AeBAD-S/train/good + 70% de normais DLR não pertencentes ao teste.",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"protocol": summary, "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return run_dir

    rtdetr_data_yaml, rtdetr_training = _materialize_rtdetr_finetune_dataset(
        protocol,
        run_dir / "rtdetr_finetune_dataset",
        train_maximum=args.rtdetr_train_images,
        validation_maximum=args.rtdetr_validation_images,
        seed=args.seed,
    )
    (run_dir / "rtdetr_execution_config.json").write_text(
        json.dumps(
            {
                "model_source": args.rtdetr_model,
                "initialization_weights": args.rtdetr_initialization_weights,
                "epochs_requested": args.rtdetr_epochs,
                "image_size": args.rtdetr_image_size,
                "batch_size": args.rtdetr_batch_size,
                "patience": args.rtdetr_patience,
                "training_data": rtdetr_training,
                "base_dataset_yaml": base_data_yaml.as_posix(),
                "fine_tune_dataset_yaml": rtdetr_data_yaml.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.skip_autoencoder:
        auto_weights, auto_metadata = _load_autoencoder_candidate(run_dir)
    else:
        auto_weights, auto_metadata = _train_autoencoder(
            protocol,
            run_dir,
            input_size=args.autoencoder_image_size,
            epochs=args.autoencoder_epochs,
            batch_size=args.autoencoder_batch_size,
            seed=args.seed,
        )
    detector_weights = _train_rtdetr(
        rtdetr_data_yaml,
        run_dir,
        model_source=args.rtdetr_model,
        initialization_weights=args.rtdetr_initialization_weights or None,
        epochs=args.rtdetr_epochs,
        image_size=args.rtdetr_image_size,
        batch_size=args.rtdetr_batch_size,
        patience=args.rtdetr_patience,
        seed=args.seed,
    )
    auto_test = _run_autoencoder_test(protocol.test, auto_weights, auto_metadata)
    detector_test = _run_rtdetr_test(
        protocol.test,
        detector_weights,
        image_size=args.rtdetr_image_size,
        confidence_threshold=args.rtdetr_confidence_threshold,
    )
    auto_metrics = _evaluate_model(protocol.test, auto_test["predictions"], float(auto_test["pixel_threshold"]))
    detector_metrics = _evaluate_model(protocol.test, detector_test["predictions"], args.rtdetr_confidence_threshold)
    results = {
        "run_dir": run_dir.as_posix(),
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": summary,
        "autoencoder": {
            "weights": auto_weights.as_posix(),
            "metadata": auto_metadata,
            "timing": auto_test["timing"],
            "metrics": auto_metrics,
        },
        "rtdetr": {
            "weights": detector_weights.as_posix(),
            "model_source": args.rtdetr_model,
            "initialization_weights": args.rtdetr_initialization_weights or None,
            "training_data": rtdetr_training,
            "input_size": args.rtdetr_image_size,
            "confidence_threshold": args.rtdetr_confidence_threshold,
            "timing": detector_test["timing"],
            "metrics": detector_metrics,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "test_predictions.json").write_text(
        json.dumps(
            {
                "autoencoder": auto_test["predictions"],
                "rtdetr": detector_test["predictions"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_confusion_matrix(
        auto_metrics["image_level"],
        run_dir / "autoencoder" / "confusion_matrix_image_level.png",
        "Autoencoder - matriz de confusao por imagem",
    )
    _save_confusion_matrix(
        detector_metrics["image_level"],
        run_dir / "rtdetr" / "confusion_matrix_image_level.png",
        "RT-DETR - matriz de confusao por imagem",
    )
    visual_result = _make_visual_examples(
        protocol.test,
        auto_test["predictions"],
        detector_test["predictions"],
        run_dir / "visual_examples",
    )
    auto_observations = _error_observations(protocol.test, auto_test["predictions"])
    detector_observations = _error_observations(protocol.test, detector_test["predictions"])
    (run_dir / "error_observations.json").write_text(
        json.dumps(
            {"autoencoder": auto_observations, "rtdetr": detector_observations, "visual_examples": visual_result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_error_study(
        run_dir / "ERROR_STUDY.md",
        results=results,
        auto_observations=auto_observations,
        rtdetr_observations=detector_observations,
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "visual_examples": visual_result,
                "autoencoder_map50": auto_metrics["map_50"],
                "rtdetr_map50": detector_metrics["map_50"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina e avalia RT-DETR DLR e autoencoder AeBAD-S+DLR sem fuga de teste."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--autoencoder-dlr-fraction", type=float, default=0.70)
    parser.add_argument("--autoencoder-image-size", type=int, default=384)
    parser.add_argument("--autoencoder-epochs", type=int, default=12)
    parser.add_argument("--autoencoder-batch-size", type=int, default=4)
    parser.add_argument(
        "--rtdetr-model",
        default="yolov8n-rtdetr.yaml",
        help="Configuração RT-DETR; o n híbrido é viável em CPU e mantém o descodificador RT-DETR.",
    )
    parser.add_argument(
        "--rtdetr-initialization-weights",
        default="yolov8n.pt",
        help="Pesos genéricos opcionais para inicializar a espinha dorsal antes do fine-tuning DLR.",
    )
    parser.add_argument("--rtdetr-epochs", type=int, default=8)
    parser.add_argument("--rtdetr-image-size", type=int, default=384)
    parser.add_argument("--rtdetr-batch-size", type=int, default=4)
    parser.add_argument("--rtdetr-patience", type=int, default=3)
    parser.add_argument(
        "--rtdetr-train-images",
        type=int,
        default=320,
        help="Máximo de imagens DLR, equilibradas por classe, usadas no fine-tuning RT-DETR (0 usa todas).",
    )
    parser.add_argument(
        "--rtdetr-validation-images",
        type=int,
        default=160,
        help="Máximo de imagens DLR, equilibradas por classe, usadas para selecionar o checkpoint RT-DETR (0 usa todas).",
    )
    parser.add_argument("--rtdetr-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-run", type=Path, help="Retoma um protocolo já preparado, sem alterar o teste selado.")
    parser.add_argument("--skip-autoencoder", action="store_true", help="Reutiliza o candidato de autoencoder já concluído.")
    args = parser.parse_args()
    if args.autoencoder_image_size % 16:
        parser.error("--autoencoder-image-size deve ser divisível por 16.")
    _run(args)


if __name__ == "__main__":
    main()
