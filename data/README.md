# Dados locais e proveniência

Esta entrega não inclui imagens, anotações, bases de dados de revisões nem ficheiros de treino. O diretório `datasets/`, criado localmente quando necessário, está ignorado por Git.

## Convenção local

Os scripts esperam, quando for legalmente aplicável, uma pasta irmã com a seguinte forma:

```text
datasets/
├── aebad/AeBAD/AeBAD_S/...
├── dlr_aircraft_dent/...
├── bladesynth/Dataset_bladesynth/...
├── agdd/data/...
├── mvtec_ad_metal_nut/metal_nut/...
├── kolektor_sdd2/extracted/...
└── neu_det/extracted/NEU-DET/...
```

Os manifestos de treino preservam `source_dataset`, rótulo original, papel experimental, indicação de aeronáutico/sintético e caminho local. Não fundem automaticamente taxonomias que apenas parecem semelhantes.

## Regras de utilização

- DLR Aircraft Dent: treino/validação/teste do detetor binário de amolgadelas.
- AeBAD-S: investigação de normalidade e taxonomia de ROI; não é prova de generalização para toda a aviação.
- BladeSynth: pré-treino sintético, nunca avaliação operacional.
- AGDD, MVTec AD, KolektorSDD2 e NEU-DET: robustez, localização ou tarefas auxiliares separadas; não definem automaticamente as classes aeronáuticas.
- Dados de revisão humana: só entram em candidatos de aprendizagem ativa após decisão final e revisão de proveniência.

Os links e notas de licenciamento estão em [`../docs/ATTRIBUTIONS.md`](../docs/ATTRIBUTIONS.md).
