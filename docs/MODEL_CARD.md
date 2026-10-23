# Cartão do modelo — AeroVision Hack2026

## Finalidade e escopo

O protótipo ajuda um inspetor a priorizar e documentar a revisão visual de imagens RGB de superfícies metálicas. A saída do modelo é um sinal de triagem local e caixas candidatas; não é um juízo de aeronavegabilidade, não deteta defeitos internos e não substitui uma inspeção autorizada.

## Modelo que participa na demonstração

| Campo | Registo |
| --- | --- |
| Modelo | RT-DETR, candidato `full_dlr_continuation_20261023` |
| Tarefa | Triagem binária normal/amolgadela e localização de `dent` no DLR Aircraft Dent |
| Entrada | Imagem RGB; inferência a 384 px |
| Saídas | Score máximo, recomendação de triagem, caixas e mapa espacial das caixas |
| Limiar operacional | 0,1786108272594388, fixado na validação DLR |
| Uso permitido nesta entrega | Demonstração offline e apoio à revisão humana |
| Uso não permitido | Aprovação/reprovação autónoma, certificação ou afirmação de cobertura de outros defeitos |

O dashboard não mistura o autoencoder com o RT-DETR. Mostrar um residual que não melhora uma decisão validada seria evidência visual enganadora. Também não usa o ResNet de ROI na inferência ativa.

## Dados e separação

| Papel | Imagens DLR |
| --- | ---: |
| Treino | 1 556 |
| Validação/limiar | 645 |
| Teste interno separado | 285 (168 normais, 117 `dent`) |

O teste não foi usado nos gradientes, seleção de checkpoint ou calibração do limiar desta continuação. Ainda assim, é um holdout dentro de um dataset público; não equivale a validação independente num ambiente de manutenção real.

## Métricas no teste interno separado

| Métrica | Valor |
| --- | ---: |
| Accuracy por imagem | 88,42% |
| Precisão `dent` | 84,43% |
| Recall `dent` | 88,03% |
| F1 `dent` | 86,19% |
| Especificidade normal | 88,69% |
| AUROC, a partir dos scores congelados | 0,958 |
| AP de anomalia, a partir dos scores congelados | 0,957 |
| mAP@0.50 | 28,03% |
| mAP@0.50:0.95 | 8,46% |
| Inferência CPU média | 0,765 s |

A precisão de localização a IoU 0,50 foi 7,14%, com recall de 76,92%, o que revela caixas adicionais/imprecisas apesar de boa triagem por imagem. A caixa é portanto uma pista para revisão, não uma localização certificada de reparação.

## Evidência visual (XAI)

O mapa apresentado é construído a partir das caixas retidas e dos respetivos scores, com suavização para legibilidade. Ele mostra a evidência espacial fornecida pelo detetor, mas não é Grad-CAM, não mostra causalidade e não prova que a região seja o defeito físico correto. O revisor pode editar ou criar caixas; essas revisões são auditadas.

## Limitações e riscos conhecidos

- Uma única classe supervisionada (`dent`): riscos, fissuras, corrosão, geometrias e acabamentos não representados podem falhar silenciosamente.
- Dados públicos, câmara e iluminação diferentes do ambiente final; reflexão especular continua a ser um risco de falso alarme/omissão.
- O limiar é operacional para este corpus, não uma probabilidade de defeito calibrada para produção.
- O score e a explicação são auxiliares; o sistema não deve bloquear a linha nem substituir NDT/inspeção humana.
- O desempenho pode variar com CPU/GPU, versão de dependências, compressão, foco e resolução de captura.

## Monitorização e melhoria contínua

O código do dashboard implementa duas filas de candidatos, ambas sem promoção automática:

1. Após 250 imagens aprovadas pelo humano, cria um candidato de normalidade com replay público e holdout humano.
2. Após 250 imagens reprovadas ou complexas, cria um candidato de ROI apenas para caixas com tipo explicitamente anotado; casos sem tipo permanecem auditados, sem rótulo inventado.

Cada candidato exige avaliação de regressão, versão, revisão humana e validação externa antes de qualquer promoção. Nesta submissão não foi promovido nenhum modelo a partir de aprendizagem ativa.

## Experiências preservadas

- Autoencoder AeBAD-S + DLR: F1 0,585, especificidade 1,19%, AUROC 0,475; não promovido.
- ResNet-18 ROI AeBAD-S: F1 macro 0,9744 em recortes de validação `view`; não é benchmark ponta-a-ponta e não é executado pelo dashboard.
- ResNet BladeSynth → AeBAD-S: F1 macro 0,9609; inferior ao ResNet base, não promovido.

Consulte os detalhes e os artefactos em [`../exploratory_models/`](../exploratory_models/).
