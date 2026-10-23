# Checkpoint RT-DETR local

O checkpoint avaliado já está incluído no repositório, em:

```text
dashboard/models/rtdetr/best.pt
```

Em alternativa, defina a variável de ambiente `AEROVISION_RTDETR_WEIGHTS` com o caminho absoluto de outro ficheiro local que pretenda testar.

## Identidade do artefacto avaliado

| Campo | Valor |
| --- | --- |
| Candidato | `full_dlr_continuation_20261023` |
| Dados de treino | 1 556 imagens DLR Aircraft Dent |
| Validação para limiar | 645 imagens DLR |
| Limiar operacional | 0,1786108272594388 |
| SHA-256 do checkpoint avaliado | `BB8E1659971634BD3C37A93D91BB802D476E27E947A0F1B219396304290BA8EF` |

O ficheiro `model_metadata.json` acompanha o checkpoint com metadados de reprodução e resultados. O hash abaixo permite verificar que `best.pt` não foi alterado.

Antes de obter, executar ou redistribuir pesos, verifique a licença do runtime Ultralytics, dos pesos de inicialização e do dataset DLR. Consulte [`../../../docs/ATTRIBUTIONS.md`](../../../docs/ATTRIBUTIONS.md).
