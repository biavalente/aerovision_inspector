from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from backend.app.config import DATA_DIR, DATASETS_DIR, MODELS_DIR  # noqa: E402
from backend.app.db import connect, init_db, now_iso  # noqa: E402
from backend.app.models.defect_type_resnet import DEFECT_TYPE_CLASSES, ROI_CLASS_NAMES, DefectTypeResNet18  # noqa: E402


INPUT_SIZE = 224
CACHE_SIZE = 288
CACHE_VERSION = "aebad_roi_mask_context_v2_normal_rejection"
DEFAULT_TRAIN_CONDITIONS = ("background", "illumination", "same")
DEFAULT_VALIDATION_CONDITIONS = ("view",)
PROMOTION_MACRO_F1 = 0.70
PROMOTION_MIN_RECALL = 0.50
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)
RESNET_WEIGHTS_PATH = MODELS_DIR / "resnet18-f37072fd.pth"
RESNET_WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
BLADESYNTH_PRETRAIN_PATH = MODELS_DIR / "bladesynth_resnet18_spatial_pretrain.pt"


@dataclass(frozen=True)
class RoiRecord:
    image_path: Path
    mask_path: Path | None
    label_index: int
    condition: str
    bbox: tuple[int, int, int, int] | None
    cache_path: Path
    normal_variant: int = 0


def _pngs(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.png")
        if path.is_file() and not path.name.startswith("._")
    )


def _bbox_from_mask(mask_path: Path) -> tuple[int, int, int, int] | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Máscara inacessível: {mask_path}")
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return int(x), int(y), int(width), int(height)


def _crop_with_context(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, box_width, box_height = bbox
    centre_x = x + box_width / 2
    centre_y = y + box_height / 2
    # Match the production crop: preserve the defect, its immediate material
    # texture and a margin for imperfect detector boxes.
    side = max(float(max(box_width, box_height)) * 2.2, float(min(height, width)) * 0.08, 192.0)
    x1 = max(0, int(round(centre_x - side / 2)))
    y1 = max(0, int(round(centre_y - side / 2)))
    x2 = min(width, int(round(centre_x + side / 2)))
    y2 = min(height, int(round(centre_y + side / 2)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("A região de defeito calculada ficou vazia.")
    return crop


def _cache_path(split: str, label: str, condition: str, image_path: Path, variant: int = 0) -> Path:
    suffix = f"__v{variant}" if variant else ""
    return DATA_DIR / "derived" / CACHE_VERSION / split / label / f"{condition}__{image_path.stem}{suffix}.jpg"


def _records_for_conditions(
    split: str,
    conditions: tuple[str, ...],
    use_existing_cache: bool,
) -> list[RoiRecord]:
    root = DATASETS_DIR / "aebad" / "AeBAD" / "AeBAD_S"
    test_root = root / "test"
    mask_root = root / "ground_truth"
    records: list[RoiRecord] = []
    for label_index, label in enumerate(DEFECT_TYPE_CLASSES, start=1):
        for condition in conditions:
            image_directory = test_root / label / condition
            if not image_directory.exists():
                raise RuntimeError(f"Condição AeBAD-S ausente: {image_directory}")
            for image_path in _pngs(image_directory):
                relative = image_path.relative_to(test_root / label)
                mask_path = mask_root / label / relative
                if not mask_path.exists():
                    raise RuntimeError(f"Máscara correspondente ausente: {mask_path}")
                cache_path = _cache_path(split, label, condition, image_path)
                if cache_path.exists() and use_existing_cache:
                    bbox = (0, 0, 0, 0)
                else:
                    bbox = _bbox_from_mask(mask_path)
                    if bbox is None:
                        continue
                records.append(
                    RoiRecord(
                        image_path=image_path,
                        mask_path=mask_path,
                        label_index=label_index,
                        condition=condition,
                        bbox=bbox,
                        cache_path=cache_path,
                    )
                )
    for condition in conditions:
        image_directory = test_root / "good" / condition
        if not image_directory.exists():
            raise RuntimeError(f"Condição AeBAD-S normal ausente: {image_directory}")
        for image_path in _pngs(image_directory):
            # Two deterministic, texture-preserving normal patches per image
            # make the rejection class resemble a detector proposal rather
            # than a full, unrelated image.
            for variant in (0, 1):
                records.append(
                    RoiRecord(
                        image_path=image_path,
                        mask_path=None,
                        label_index=0,
                        condition=condition,
                        bbox=None,
                        cache_path=_cache_path(split, "normal_roi", condition, image_path, variant),
                        normal_variant=variant,
                    )
                )
    return records


def _prepare_cache(records: Iterable[RoiRecord], rebuild: bool) -> int:
    written = 0
    for record in records:
        if record.cache_path.exists() and not rebuild:
            continue
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Imagem inacessível: {record.image_path}")
        if record.bbox is None:
            crop = _normal_context_crop(image, record.image_path, record.normal_variant)
        else:
            crop = _crop_with_context(image, record.bbox)
        crop = cv2.resize(crop, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
        record.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(record.cache_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise OSError(f"Não foi possível criar o recorte: {record.cache_path}")
        written += 1
    return written


def _normal_context_crop(image: np.ndarray, image_path: Path, variant: int) -> np.ndarray:
    """Create a deterministic normal material patch with defect-like context."""
    height, width = image.shape[:2]
    digest = hashlib.sha256(f"{image_path.as_posix()}::{variant}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], byteorder="big"))
    side = min(height, width) * generator.uniform(0.18, 0.42)
    centre_x = width * generator.uniform(0.28, 0.72)
    centre_y = height * generator.uniform(0.28, 0.72)
    x1 = max(0, int(round(centre_x - side / 2)))
    y1 = max(0, int(round(centre_y - side / 2)))
    x2 = min(width, int(round(centre_x + side / 2)))
    y2 = min(height, int(round(centre_y + side / 2)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("A região normal calculada ficou vazia.")
    return crop


def _augment(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    if random.random() < 0.5:
        angle = random.uniform(-6.0, 6.0)
        matrix = cv2.getRotationMatrix2D((CACHE_SIZE / 2, CACHE_SIZE / 2), angle, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (CACHE_SIZE, CACHE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    if random.random() < 0.75:
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-20, 20)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.20:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=random.uniform(0.25, 0.9))
    scale = random.uniform(0.82, 1.0)
    crop_size = max(INPUT_SIZE, int(CACHE_SIZE * scale))
    maximum_offset = CACHE_SIZE - crop_size
    x = random.randint(0, maximum_offset)
    y = random.randint(0, maximum_offset)
    image = image[y : y + crop_size, x : x + crop_size]
    return cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def _to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_rgb.copy()).permute(2, 0, 1)


class CachedRoiDataset(Dataset):
    def __init__(self, records: list[RoiRecord], augment: bool) -> None:
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = cv2.imread(str(record.cache_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Recorte inacessível: {record.cache_path}")
        if self.augment:
            image = _augment(image)
        else:
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        return _to_tensor(image), torch.tensor(record.label_index, dtype=torch.long)


def _metrics(targets: list[int], predictions: list[int], mean_loss: float) -> tuple[dict[str, float], list[list[int]]]:
    class_count = len(ROI_CLASS_NAMES)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1
    total = max(1, len(targets))
    accuracy = sum(confusion[index][index] for index in range(class_count)) / total
    recalls: list[float] = []
    f1_scores: list[float] = []
    result: dict[str, float] = {
        "validation_accuracy": round(float(accuracy), 4),
        "validation_loss": round(float(mean_loss), 4),
    }
    for index, label in enumerate(ROI_CLASS_NAMES):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(class_count) if row != index)
        false_negative = sum(confusion[index][column] for column in range(class_count) if column != index)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        result[f"precision_{label}"] = round(float(precision), 4)
        result[f"recall_{label}"] = round(float(recall), 4)
        result[f"f1_{label}"] = round(float(f1), 4)
        recalls.append(recall)
        f1_scores.append(f1)
    result["balanced_recall"] = round(float(np.mean(recalls)), 4)
    result["macro_f1"] = round(float(np.mean(f1_scores)), 4)
    result["minimum_class_recall"] = round(float(min(recalls)), 4)
    return result, confusion


def _evaluate(
    model: DefectTypeResNet18,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[dict[str, float], list[list[int]], torch.Tensor, torch.Tensor]:
    model.eval()
    targets_all: list[int] = []
    predictions_all: list[int] = []
    logits_all: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            losses.append(float(loss_fn(logits, targets).item()))
            predictions = logits.argmax(dim=1)
            targets_all.extend(targets.tolist())
            predictions_all.extend(predictions.tolist())
            logits_all.append(logits.cpu())
    metrics, confusion = _metrics(targets_all, predictions_all, float(np.mean(losses)))
    return metrics, confusion, torch.cat(logits_all), torch.tensor(targets_all, dtype=torch.long)


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in np.linspace(0.55, 3.0, 50):
        loss = float(nn.functional.cross_entropy(logits / float(temperature), targets).item())
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = loss
    return round(best_temperature, 3)


def _split_conditions(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Indique pelo menos uma condição AeBAD-S.")
    return result


def _register_model(
    version_name: str,
    status: str,
    metrics: dict[str, float],
    train_records: list[RoiRecord],
    validation_records: list[RoiRecord],
    note: str,
) -> None:
    init_db()
    with connect() as connection:
        if status == "active_type_support":
            connection.execute(
                """
                UPDATE model_versions
                SET status = 'baseline_type_support'
                WHERE model_type = 'roi_defect_type_resnet18' AND status = 'active_type_support'
                """
            )
        connection.execute(
            """
            INSERT INTO model_versions (
                version_name, model_type, status, created_at, dataset_snapshot_json, metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_name) DO UPDATE SET
                status=excluded.status,
                metrics_json=excluded.metrics_json,
                notes=excluded.notes
            """,
            (
                version_name,
                "roi_defect_type_resnet18",
                status,
                now_iso(),
                json.dumps(
                    {
                        "dataset": "AeBAD-S/test local development corpus",
                        "train_count": len(train_records),
                        "validation_count": len(validation_records),
                        "train_conditions": sorted({record.condition for record in train_records}),
                        "validation_conditions": sorted({record.condition for record in validation_records}),
                        "roi_supervision": "ground-truth masks used only to create training and validation crops",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(metrics, ensure_ascii=False),
                note,
            ),
        )
        connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina localmente um classificador de tipo de defeito em recortes AeBAD-S anotados."
    )
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-conditions", type=_split_conditions, default=DEFAULT_TRAIN_CONDITIONS)
    parser.add_argument("--validation-conditions", type=_split_conditions, default=DEFAULT_VALIDATION_CONDITIONS)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--backbone-init",
        choices=("imagenet", "bladesynth"),
        default="imagenet",
        help="Inicialização da espinha dorsal; BladeSynth é pré-treino espacial genérico, não taxonomia final.",
    )
    parser.add_argument("--bladesynth-checkpoint", type=Path, default=BLADESYNTH_PRETRAIN_PATH)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promove explicitamente um candidato que passe as portas internas. Sem esta opção, fica em validação.",
    )
    args = parser.parse_args()

    overlapping = set(args.train_conditions) & set(args.validation_conditions)
    if overlapping:
        raise RuntimeError(f"As condições de treino e validação não podem sobrepor-se: {sorted(overlapping)}")
    if args.backbone_init == "imagenet" and not RESNET_WEIGHTS_PATH.exists():
        raise RuntimeError(
            "Pesos locais ResNet-18 ausentes. Descarregue uma vez os pesos oficiais antes do treino: "
            f"{RESNET_WEIGHTS_URL}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    train_records = _records_for_conditions("train", args.train_conditions, use_existing_cache=not args.rebuild_cache)
    validation_records = _records_for_conditions(
        "validation",
        args.validation_conditions,
        use_existing_cache=not args.rebuild_cache,
    )
    if not train_records or not validation_records:
        raise RuntimeError("Não foi possível construir os recortes de treino e validação.")
    train_cache_written = _prepare_cache(train_records, args.rebuild_cache)
    validation_cache_written = _prepare_cache(validation_records, args.rebuild_cache)
    print(
        json.dumps(
            {
                "train_samples": len(train_records),
                "validation_samples": len(validation_records),
                "train_cache_written": train_cache_written,
                "validation_cache_written": validation_cache_written,
                "train_conditions": args.train_conditions,
                "validation_conditions": args.validation_conditions,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    class_counts = Counter(record.label_index for record in train_records)
    sampling_weights = torch.DoubleTensor([1.0 / class_counts[record.label_index] for record in train_records])
    sampler = WeightedRandomSampler(sampling_weights, num_samples=len(train_records), replacement=True)
    train_loader = DataLoader(
        CachedRoiDataset(train_records, augment=True),
        batch_size=max(1, args.batch_size),
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        CachedRoiDataset(validation_records, augment=False),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    model = DefectTypeResNet18()
    if args.backbone_init == "bladesynth":
        if not args.bladesynth_checkpoint.exists():
            raise RuntimeError(
                "Checkpoint de pré-treino BladeSynth ausente. Execute primeiro scripts/pretrain_resnet_bladesynth.py "
                "ou indique --bladesynth-checkpoint."
            )
        model.load_backbone_checkpoint(args.bladesynth_checkpoint)
        backbone_provenance = {
            "architecture": "ResNet-18",
            "weights_source": "candidate spatial pretraining on BladeSynth masks",
            "weights_file": args.bladesynth_checkpoint.name,
            "normalization": "ImageNet mean/std",
        }
    else:
        model.load_imagenet_backbone(RESNET_WEIGHTS_PATH)
        backbone_provenance = {
            "architecture": "ResNet-18",
            "weights_source": RESNET_WEIGHTS_URL,
            "weights_file": RESNET_WEIGHTS_PATH.name,
            "normalization": "ImageNet mean/std",
        }
    backbone_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": 1.5e-4},
            {"params": model.fc.parameters(), "lr": 1.0e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.04)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_confusion: list[list[int]] | None = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets in train_loader:
            logits = model(images)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        metrics, confusion, _logits, _targets = _evaluate(model, validation_loader, loss_fn)
        improved = best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_confusion = confusion
            patience = 0
        else:
            patience += 1
        print(
            f"epoch {epoch}/{args.epochs}: train_loss={np.mean(losses):.4f} "
            f"val_macro_f1={metrics['macro_f1']:.3f} val_accuracy={metrics['validation_accuracy']:.3f}",
            flush=True,
        )
        if patience >= 5:
            print("early_stop: 5 épocas sem melhoria de macro-F1", flush=True)
            break

    if best_state is None or best_metrics is None or best_confusion is None:
        raise RuntimeError("O treino não produziu um estado de modelo válido.")
    model.load_state_dict(best_state)
    final_metrics, final_confusion, validation_logits, validation_targets = _evaluate(model, validation_loader, loss_fn)
    temperature = _fit_temperature(validation_logits, validation_targets)
    final_metrics["temperature"] = temperature
    final_metrics["best_epoch_macro_f1"] = best_metrics["macro_f1"]
    version_name = f"roi_defect_type_resnet18_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    passed_internal_gate = (
        final_metrics["macro_f1"] >= PROMOTION_MACRO_F1
        and final_metrics["minimum_class_recall"] >= PROMOTION_MIN_RECALL
    )
    status = "active_type_support" if passed_internal_gate and args.promote else (
        "validation_required" if passed_internal_gate else "validation_failed"
    )
    notes = (
        "Classificador local de tipo de defeito em recortes de ROI, com uma classe normal de rejeição. "
        "O treino usa máscaras AeBAD-S apenas para gerar os recortes defeituosos; durante a inferência, a ROI "
        "vem de uma varredura local multi-escala, sem máscara. "
        "A validação retém a condição de captura 'view', separada das condições de treino. "
        "Não é benchmark oficial nem prova de generalização para aeronaves operacionais."
    )
    if status == "active_type_support":
        notes += " Passou os limiares internos para apoio não bloqueante à revisão, com desacordo entre tipos encaminhado como caso complexo."
    elif status == "validation_required":
        notes += " Passou as portas internas, mas fica como candidato até validação humana e promoção explícita."
    else:
        notes += " Não passou os limiares internos e não é usado no fluxo de inspeção."

    metadata = {
        "version_name": version_name,
        "model_type": "roi_defect_type_resnet18",
        "trained_at": now_iso(),
        "input_size": INPUT_SIZE,
        "class_names": list(ROI_CLASS_NAMES),
        "defect_type_classes": list(DEFECT_TYPE_CLASSES),
        "training_dataset": "AeBAD-S/test local development corpus",
        "training_conditions": list(args.train_conditions),
        "validation_conditions": list(args.validation_conditions),
        "training_image_count": len(train_records),
        "validation_image_count": len(validation_records),
        "epochs_requested": args.epochs,
        "inconclusive_confidence_threshold": 0.45,
        "promotion_status": status,
        "promotion_gate": {
            "macro_f1_minimum": PROMOTION_MACRO_F1,
            "minimum_class_recall": PROMOTION_MIN_RECALL,
        },
        "metrics": final_metrics,
        "confusion_matrix": final_confusion,
        "roi_supervision": (
            "A máscara de ground truth é usada exclusivamente para construir os recortes de treino e validação. "
            "A inferência examina ROIs de uma varredura local multi-escala; recortes normais são gerados apenas "
            "a partir de imagens AeBAD-S good."
        ),
        "pretrained_backbone": backbone_provenance,
        "promotion_requested": bool(args.promote),
        "limitations": notes,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_weights = MODELS_DIR / f"{version_name}.pt"
    candidate_metadata = MODELS_DIR / f"{version_name}.json"
    torch.save({"state_dict": model.state_dict()}, candidate_weights)
    candidate_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if status == "active_type_support":
        shutil.copy2(candidate_weights, MODELS_DIR / "defect_type_resnet18.pt")
        shutil.copy2(candidate_metadata, MODELS_DIR / "defect_type_resnet18.json")
    _register_model(version_name, status, final_metrics, train_records, validation_records, notes)
    print(
        json.dumps(
            {
                "version": version_name,
                "status": status,
                "candidate_weights": str(candidate_weights),
                **final_metrics,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
