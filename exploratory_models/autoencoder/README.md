# Exploração — autoencoder de normalidade

## Porque foi tentado

Um autoencoder pode aprender a aparência de peças normais e usar erro de reconstrução como sinal de anomalia. É uma hipótese útil quando há poucos defeitos anotados, novos tipos de falha ou necessidade de um sinal complementar de “fora do padrão”. Por isso foi estudado antes de se escolher um detetor supervisionado como saída principal.

## Experiência DLR + AeBAD-S

O candidato `dlr_mixed_autoencoder` foi treinado durante 12 épocas, a 384 px, com 1 694 imagens normais: 521 AeBAD-S `train/good` e 1 173 normais DLR. O teste separado DLR tinha 285 imagens, nunca vistas no treino: 168 normais e 117 com amolgadela.

| Medida | Resultado |
| --- | ---: |
| Accuracy por imagem | 41,75% |
| Precisão (`dent`) | 41,34% |
| Recall (`dent`) | 100,00% |
| F1 (`dent`) | 58,50% |
| Especificidade normal | 1,19% |
| AUROC | 0,475 |
| AP de anomalia | 0,434 |
| mAP@0.50 residual exploratório | 0,00% |
| Inferência CPU média | 0,219 s |

O resultado mostra que este residual, nesta configuração, classifica quase todas as peças normais como anómalas. Por isso o candidato **não foi promovido** nem é executado pelo dashboard atual. Manter este resultado no repositório é intencional: evita alegar uma capacidade que os dados não sustentam.

## Conteúdo

- `scripts/`: treino, avaliação, recalibração, fine-tuning por imagens normais humanas e geração comparativa.
- `results/dlr_mixed_autoencoder_metadata.json`: configuração e histórico do candidato DLR.
- `results/confusion_matrix_image_level.png`: matriz do candidato no teste DLR.
- `results/DLR_COMPARISON.md`, CSV e gráficos: comparação congelada contra RT-DETR.
- `results/comparison_examples/`: 20 painéis com referência, previsão autoencoder e previsão RT-DETR.

## Próximo passo responsável

Uma nova abordagem de anomalia (por exemplo PatchCore/Anomalib, melhor modelação de iluminação ou dados de câmara final) deve ser criada como candidato separado, avaliada em validação e teste externo, e só depois considerada para combinação com um detetor. Não se deve voltar a apresentar o mapa residual atual como evidência de decisão no dashboard.
