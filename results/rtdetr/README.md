# Resultados RT-DETR — candidato usado na demonstração

Esta pasta contém os artefactos do candidato `full_dlr_continuation_20261023`, treinado exclusivamente para localizar a classe `dent`/amolgadela do DLR Aircraft Dent. Não contém pesos nem imagens de treino.

## Protocolo registado

- Treino: 1 556 imagens DLR (934 normais, 622 com `dent`).
- Validação para seleção de limiar: 645 imagens DLR (369 normais, 276 com `dent`).
- Teste interno separado: 285 imagens DLR (168 normais, 117 com `dent`).
- Limiar: `0,178611`, escolhido só na validação antes da avaliação final.
- Hardware de medição: execução local em CPU; tempos não devem ser extrapolados para outro computador.

## Métricas do teste separado

| Medida | Resultado |
| --- | ---: |
| Matriz por imagem | TN 149 · FP 19 · FN 14 · TP 103 |
| Accuracy | 88,42% |
| Precisão (`dent`) | 84,43% |
| Recall (`dent`) | 88,03% |
| F1 (`dent`) | 86,19% |
| Especificidade (`normal`) | 88,69% |
| AUROC | 0,958 |
| AP de anomalia | 0,957 |
| mAP@0.50 | 28,03% |
| mAP@0.50:0.95 | 8,46% |
| Precisão de localização a IoU 0,50 | 7,14% |
| Recall de localização a IoU 0,50 | 76,92% |
| Inferência CPU média | 0,765 s |
| Inferência CPU mínimo–máximo | 0,284–1,437 s |

`mAP` avalia as caixas; as métricas por imagem avaliam se a imagem foi encaminhada como normal ou com amolgadela. Não são a mesma medida. A diferença entre recall de localização e precisão de localização também evidencia propostas de caixas adicionais/imprecisas, razão pela qual a caixa deve ser revista por um humano.

## Conteúdo

- `metrics.json`: resumo legível por máquina, sem caminhos da máquina que gerou o ensaio.
- `confusion_matrix_image_level.png`: matriz de confusão por imagem.
- `ERROR_STUDY.md`: hipóteses e leitura dos erros.
- `visual_examples/`: vinte painéis de referência vs. previsão — cinco erros e quinze previsões corretas — mais o manifesto de seleção.

As imagens são exemplos de avaliação para apresentação; continuam sujeitas às condições do dataset de origem. Consulte [`../../docs/ATTRIBUTIONS.md`](../../docs/ATTRIBUTIONS.md).
