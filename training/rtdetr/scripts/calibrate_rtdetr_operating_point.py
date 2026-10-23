"""Calibrate a RT-DETR decision threshold without touching the sealed test set.

This post-training utility deliberately uses the selected DLR fine-tuning
validation split to choose a binary *image* operating point.  It then applies
that frozen threshold to the already-exported sealed-test predictions.  The
detector's AP/mAP is recomputed from all scored boxes and therefore remains
threshold-independent.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import run_dlr_benchmark as benchmark


def _selection_records(selection_path: Path) -> tuple[benchmark.DlrRecord, ...]:
    records: list[benchmark.DlrRecord] = []
    with selection_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "rtdetr_finetune_validation":
                continue
            records.append(
                benchmark.DlrRecord(
                    sample_id=str(row["sample_id"]),
                    official_split=str(row["sample_id"]).split("/", 1)[0],
                    image_path=Path(str(row["image_path"])),
                    mask_path=Path(str(row["mask_path"])),
                    has_dent=str(row["ground_truth"]) == "dent",
                )
            )
    if not records:
        raise RuntimeError(f"Não foram encontradas imagens de validação em {selection_path}.")
    return tuple(sorted(records, key=lambda item: item.sample_id))


def _with_threshold(
    predictions: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    updated = copy.deepcopy(predictions)
    for prediction in updated.values():
        prediction["image_prediction_dent"] = bool(float(prediction["score"]) >= threshold)
    return updated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _calibrated_error_study(
    calibration: dict[str, Any],
    visual_result: dict[str, Any],
) -> str:
    metrics = calibration["sealed_test"]["metrics"]
    image = metrics["image_level"]
    detection = metrics["detection_level_iou_0_50"]
    validation = calibration["calibration_split"]["metrics"]
    threshold = calibration["selected_confidence_threshold"]
    return f"""# Estudo de erros — RT-DETR calibrado — benchmark DLR Aircraft Dent

## Protocolo

- O checkpoint RT-DETR foi treinado apenas com uma seleção balanceada DLR: 320 imagens para gradientes (160 normais, 160 dents) e 160 imagens DLR de validação (80/80).
- Os pesos YOLOv8n públicos serviram apenas para inicialização visual genérica; não houve treino supervisionado externo de defeitos.
- O limiar operacional `{threshold:.6f}` foi escolhido **somente** nas 160 imagens de validação, maximizando F1 de imagem dent. O teste final não participou nessa escolha.
- O teste selado tem 285 imagens DLR: 168 normais e 117 com dent.

## Resultado da validação usada para calibrar

| Métrica | Valor |
| --- | ---: |
| Accuracy | {validation['accuracy']:.4f} |
| Precisão dent | {validation['precision_dent']:.4f} |
| Recall dent | {validation['recall_dent']:.4f} |
| F1 dent | {validation['f1_dent']:.4f} |

## Resultado no teste selado

| Nível | Métrica | Valor |
| --- | --- | ---: |
| Imagem | Accuracy | {image['accuracy']:.4f} |
| Imagem | Precisão dent | {image['precision_dent']:.4f} |
| Imagem | Recall dent | {image['recall_dent']:.4f} |
| Imagem | F1 dent | {image['f1_dent']:.4f} |
| Localização | Precisão @IoU 0.50 | {detection['precision']:.4f} |
| Localização | Recall @IoU 0.50 | {detection['recall']:.4f} |
| Localização | mAP@0.50 | {metrics['map_50']:.4f} |
| Localização | mAP@0.50:0.95 | {metrics['map_50_95']:.4f} |

## Leitura correta destes números

- O classificador ao nível da imagem aprendeu um sinal útil (F1 {image['f1_dent']:.3f}), mas a localização ainda é fraca (mAP@0.50 {metrics['map_50']:.3f}). Por isso, não deve aprovar/reprovar uma peça autonomamente.
- Há {int(image['false_positive'])} falsos positivos e {int(image['false_negative'])} falsos negativos por imagem. Estes casos devem alimentar a revisão humana e as filas de active learning; não devem ser escondidos.
- A precisão muito baixa das caixas resulta de muitas propostas de baixa confiança ao usar um limiar baixo. O limiar é adequado para triagem de imagem, não é prova de localização precisa.
- As 20 imagens de demonstração foram refeitas com este limiar: {visual_result['created']} imagens, com {visual_result['available_errors']} erros disponíveis e objetivo de {visual_result['requested_errors']} erros mostrados.

## Hipóteses verificáveis para melhorar

1. A dent ocupa pouca área e perde definição aos 384 px; treinar a 640 px com GPU e mais épocas deve melhorar fronteiras.
2. O conjunto CPU-feasible de 320 imagens é deliberadamente pequeno; aumentar o DLR de treino mantendo o teste selado é a melhoria mais direta.
3. Reflexos e textura metálica produzem propostas espúrias. Aquisição com luz anelar quadranteada e dados reais dessa câmara são mais valiosos do que ajustar apenas o limiar.
4. O dashboard deve apresentar a caixa RT-DETR como evidência para o humano, juntamente com residual do autoencoder e Grad-CAM/atenção, nunca como decisão final certificada.
"""


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    result_path = run_dir / "results.json"
    selection_path = run_dir / "rtdetr_finetune_selection.csv"
    prediction_path = run_dir / "test_predictions.json"
    protocol_path = run_dir / "protocol_manifest.csv"
    if not result_path.exists() or not selection_path.exists() or not prediction_path.exists():
        raise RuntimeError("O diretório indicado não contém um benchmark RT-DETR concluído.")

    results = json.loads(result_path.read_text(encoding="utf-8"))
    detector = results["rtdetr"]
    validation_records = _selection_records(selection_path)
    protocol = benchmark._load_protocol_manifest(protocol_path)
    image_size = int(detector["input_size"])
    validation_run = benchmark._run_rtdetr_test(
        validation_records,
        Path(detector["weights"]),
        image_size=image_size,
        confidence_threshold=1.1,
    )
    validation_predictions = validation_run["predictions"]
    normal_scores = [
        float(validation_predictions[record.sample_id]["score"])
        for record in validation_records
        if not record.has_dent
    ]
    dent_scores = [
        float(validation_predictions[record.sample_id]["score"])
        for record in validation_records
        if record.has_dent
    ]
    threshold, validation_metrics = benchmark._choose_image_threshold(normal_scores, dent_scores)

    all_stored_predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    stored_predictions = all_stored_predictions["rtdetr"]
    test_predictions = _with_threshold(stored_predictions, threshold)
    test_metrics = benchmark._evaluate_model(protocol.test, test_predictions, threshold)
    calibration = {
        "status": "calibrated_on_validation_only",
        "selected_confidence_threshold": threshold,
        "selection_rule": "maximizar F1 de imagem dent; desempate por especificidade e menor limiar",
        "calibration_split": {
            "source": "rtdetr_finetune_validation",
            "images": len(validation_records),
            "normals": sum(not record.has_dent for record in validation_records),
            "dents": sum(record.has_dent for record in validation_records),
            "metrics": validation_metrics,
            "score_range": {
                "normal_min": min(normal_scores),
                "normal_max": max(normal_scores),
                "dent_min": min(dent_scores),
                "dent_max": max(dent_scores),
            },
        },
        "sealed_test": {
            "images": len(protocol.test),
            "metrics": test_metrics,
            "note": "O teste foi apenas aplicado depois de fixar o limiar na validação; mAP usa todas as caixas classificadas e não depende deste limiar operacional.",
        },
    }
    visual_result = benchmark._make_visual_examples(
        protocol.test,
        all_stored_predictions["autoencoder"],
        test_predictions,
        run_dir / "visual_examples_calibrated",
    )
    calibration["visual_examples_calibrated"] = visual_result
    destination = run_dir / "rtdetr_operating_point.json"
    destination.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    benchmark._save_confusion_matrix(
        test_metrics["image_level"],
        run_dir / "rtdetr" / "confusion_matrix_calibrated_image_level.png",
        f"RT-DETR - teste selado - limiar {threshold:.4f}",
    )
    results["rtdetr"]["calibrated_operating_point"] = calibration
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    all_stored_predictions["rtdetr_calibrated_operating_point"] = test_predictions
    prediction_path.write_text(json.dumps(all_stored_predictions, indent=2), encoding="utf-8")
    (run_dir / "ERROR_STUDY_CALIBRATED.md").write_text(
        _calibrated_error_study(calibration, visual_result), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "threshold": threshold,
                "validation_f1": validation_metrics["f1_dent"],
                "test_image_metrics": test_metrics["image_level"],
                "test_map50": test_metrics["map_50"],
                "output": destination.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
