# Resultados comparativos — Autoencoder vs. RT-DETR

## Protocolo

- **Teste DLR selado:** 285 imagens (168 normais; 117 com amolgadela).
- **Sem fuga de dados:** as imagens deste teste não foram usadas no treino nem na calibração de limiar.
- **Reprodutibilidade:** esta pasta recalcula as métricas a partir das previsões congeladas do teste; não retreina modelos nem ajusta limiares.
- **Limiar do autoencoder:** 0.026854 no score de reconstrução; as caixas residuais usam 0.149480.
- **Limiar RT-DETR:** 0.178611 na confiança máxima da caixa.

## Métricas por imagem e localização

| Métrica | Autoencoder | RT-DETR |
| --- | ---: | ---: |
| Accuracy por imagem | 41.75% | 88.42% |
| Precisão (dent) | 41.34% | 84.43% |
| Recall (dent) | 100.00% | 88.03% |
| F1 (dent) | 58.50% | 86.19% |
| Especificidade normal | 1.19% | 88.69% |
| AUROC | 0.475 | 0.958 |
| AP de anomalia | 0.434 | 0.957 |
| mAP@0.50 (IoU 50%) | 0.00% | 28.03% |
| mAP@0.50:0.95 | 0.00% | 8.46% |
| Tempo médio / imagem | 0.219 s | 0.765 s |
| Tempo mínimo–máximo | 0.136–0.604 s | 0.284–1.437 s |

### Matrizes de confusão por imagem

| Modelo | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Autoencoder | 2 | 166 | 0 | 117 |
| RT-DETR | 149 | 19 | 14 | 103 |

## Leitura correcta dos resultados

O **autoencoder** é um modelo de normalidade: o seu score é erro de reconstrução e as suas caixas são componentes de residual. Assim, o seu `mAP@0.50` é uma medição exploratória dessas regiões residuais, **não** uma alegação de deteção supervisionada equivalente a RT-DETR.

O **RT-DETR** é o detetor supervisionado de amolgadelas. Para a apresentação, usa a sua métrica por imagem (F1/precisão/recall) para explicar o encaminhamento da peça e `mAP@0.50` para explicar a qualidade da localização. Nenhum dos dois resultados representa certificação aeronáutica ou desempenho em campo.

## Ficheiros desta pasta

- `summary.json`: resultados completos, caminhos de origem e limiares congelados.
- `metrics_by_model.csv`: tabela pronta para folha de cálculo.
- `per_image_predictions.csv`: todas as 285 previsões por imagem.
- `autoencoder_confusion_matrix.png` e `rtdetr_confusion_matrix.png`: matrizes de confusão.
- `roc_comparativa.png`: curvas ROC e AUROC.
- `20_exemplos/`: 20 imagens, cada uma com referência, previsão do autoencoder e previsão RT-DETR.
