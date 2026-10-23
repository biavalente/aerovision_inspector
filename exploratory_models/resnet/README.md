# Exploração — ResNet-18 por regiões de interesse (ROI)

## Objetivo

Este ramo estudou um classificador de tipos de defeito em recortes de regiões de interesse, não um detetor ponta-a-ponta de uma fotografia de peça. As máscaras públicas AeBAD-S foram usadas apenas para construir recortes de treino/validação; em inferência, a abordagem prevista seria varrer ROIs multi-escala sem usar máscara de referência.

## Resultado base

O candidato `roi_defect_type_resnet18_20261021T215816Z` foi treinado com 1 665 ROIs e validado em 464 ROIs da condição de captura `view`, que ficou fora do treino. Classes: `normal_roi`, `ablation`, `breakdown`, `fracture` e `groove`.

| Medida de validação ROI | Resultado |
| --- | ---: |
| Accuracy | 97,63% |
| F1 macro | 0,9744 |
| Balanced recall | 0,9731 |
| Menor recall por classe | 0,9268 |

São métricas de recortes num corpus público de desenvolvimento, não mAP de deteção ponta-a-ponta, nem desempenho de aeronaves operacionais. Por isso o ResNet foi preservado como investigação e não participa na inferência atual do dashboard.

## Pré-treino BladeSynth

Foi também testado pré-treino espacial binário sintético (4 250 ROIs de treino e 750 de validação). O resultado sintético foi 100%, mas não pode ser interpretado como validação real. Após fine-tuning AeBAD-S, o candidato obteve F1 macro 0,9609 e menor recall 0,8537 — inferior ao modelo base — e ficou em `validation_required`.

## Conteúdo

- `scripts/train_defect_type_classifier.py`: classificador ROI AeBAD-S.
- `scripts/pretrain_resnet_bladesynth.py`: pré-treino sintético auditável.
- `scripts/finetune_from_reviews.py`: candidato de aprendizagem ativa a partir de caixas humanas tipificadas.
- `scripts/probe_roi_grid.py`: ferramenta de inspeção da varredura de ROIs.
- `results/`: metadados, métricas por classe e matrizes de confusão serializadas dos candidatos.

Os pesos não são incluídos. As classes não devem ser confundidas com a classe binária `dent` do DLR ou com defeitos de outros datasets sem uma taxonomia e validação próprias.
