# Politica de decisao humana

## Estados produzidos pelo sistema

O motor de IA pode classificar uma analise como `likely_normal`,
`likely_anomaly`, `ambiguous` ou `capture_quality_warning`. Estes estados nao
param o fluxo nem libertam uma peca para servico.

## Estados finais de revisao

Um utilizador decide individualmente ou em lote entre:

- `approved_visual`: sem anomalia visual confirmada pelo revisor;
- `rejected_visual`: defeito visual confirmado;
- `escalated_ndt`: necessita de outro metodo de ensaio nao destrutivo;
- `recapture_requested`: evidencia visual insuficiente.

Uma decisao em lote altera apenas o estado final e a justificacao comum. As
anotacoes espaciais permanecem ligadas a cada imagem e sao revistas de forma
individual.

## Politica de melhoria continua

Só inspeções revistas e validadas entram no conjunto de melhoria. Há dois
contadores independentes de 250 decisões humanas:

- 250 imagens aprovadas como normais criam um candidato de autoencoder, usando
  apenas normais e uma reserva humana de validação;
- 250 imagens reprovadas ou escaladas como complexas criam um candidato de ROI,
  mas só caixas com tipo explicitamente escolhido pelo inspetor recebem rótulo
  supervisionado.

Nenhum destes ciclos substitui automaticamente o modelo ativo. A promoção exige
comparação contra um conjunto de teste bloqueado, validação externa adequada e
aprovação humana.

