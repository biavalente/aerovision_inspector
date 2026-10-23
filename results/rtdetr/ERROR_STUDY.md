# Estudo de erros — RT-DETR DLR, continuação completa

## Âmbito verificável

Este estudo refere-se ao checkpoint
`rtdetr/full_dlr_continuation_20261023/weights/best.pt`. O treino utilizou 1
556 imagens DLR (934 normais e 622 com dent); o limiar de confiança
0,1786108273 foi escolhido unicamente nas 645 imagens de validação. As 285
imagens de teste ficaram seladas até à avaliação final.

## Resultado no teste selado

Na decisão por imagem, a matriz foi: TN=149, FP=19, FN=14 e TP=103. Isto dá
accuracy 88,42%, precisão 84,43%, recall 88,03%, F1 86,19% e especificidade
88,69%. O tempo de inferência em CPU foi 0,765 s em média (mínimo 0,284 s;
máximo 1,437 s).

Na localização, a avaliação por caixas obteve 100 TP, 1 300 FP e 30 FN a
IoU=0,50: precisão por caixa 7,14% e recall 76,92%. O AP decresce de 28,03%
a IoU=0,50 para 19,20% a 0,55, 9,60% a 0,65, 3,44% a 0,75, 1,68% a 0,80 e 0%
a 0,90/0,95. O mAP@0.50:0.95 é 8,46%.

## Leitura dos erros

- Os 19 falsos positivos de imagem são peças normais que foram encaminhadas
  para revisão. Num fluxo humano-no-loop, aumentam carga do inspetor; não
  devem ser tratados como reprovação automática.
- Os 14 falsos negativos de imagem são dents que não foram suficientemente
  pontuadas. Representam o risco mais relevante e justificam manter a revisão
  e o mapa de evidência, em vez de usar o resultado como aprovação autónoma.
- A diferença entre o F1 elevado por imagem e a precisão baixa por caixa não é
  uma contradição: basta uma proposta sobreposta para marcar a imagem como
  suspeita, mas propostas adicionais contam como falsos positivos na métrica de
  deteção. Portanto, o candidato é atualmente melhor a *encaminhar* uma imagem
  do que a desenhar uma delimitação metrológica exata.
- A queda rápida de AP em IoU elevado é evidência direta de caixas com extensão
  ou posição aproximada. Não deve ser usada uma caixa prevista para medir a
  dimensão física de um dent.

## Hipóteses a verificar em recolha futura

As causas seguintes são hipóteses técnicas consistentes com as métricas; não
foram provadas individualmente por rótulos de erro adicionais:

1. Dents pequenos ou de contraste baixo perdem detalhe ao redimensionar a 384
   px, o que pode explicar parte dos falsos negativos e caixas pouco precisas.
2. Reflexos especulares, curvatura e marcas superficiais podem parecer bordos
   de dents e produzir propostas extra em peças normais.
3. As caixas derivadas do rótulo disponível podem não coincidir exatamente com
   a fronteira visual de uma deformação suave, penalizando IoU alto.
4. O desequilíbrio espacial entre fundo e área de defeito favorece muitas
   propostas de fundo, visível nos 1 300 FP de caixas.

## Decisão de uso

O checkpoint substitui o candidato DLR anterior **apenas como candidato de
demonstração/triagem com revisão humana**: melhorou mAP@0.50 de 1,01% para
28,03% e F1 por imagem de 0,6768 para 0,8619. Antes de promoção operacional,
é necessário repetir a avaliação num conjunto bloqueado da câmara e iluminação
finais, incluindo casos normais representativos e defeitos anotados por um
especialista.
