# Plano de dados e candidatos de treino

Cada dataset tem uma finalidade explícita. Os manifestos em `dashboard/data/manifests/`
guardam a proveniência, o split original, a modalidade e a regra de utilização;
eles não copiam imagens nem transformam rótulos por semelhança de nome.

## Modelo primário de normalidade

- **AeBAD-S:** normais reais de pás em cenário estático; base e replay do
  autoencoder.
- **Imagens reais aprovadas pelo inspetor:** entram no candidato primário apenas
  depois de 250 aprovações finais; nunca entram imagens defeituosas.
- **AeBAD-V:** perfil separado para vídeo, a usar apenas quando a solução tiver
  sequência temporal. Não é misturado com a calibração estática.

## Classificador visual secundário

- **AeBAD-S:** mantém a taxonomia de quatro tipos visuais já usada pela
  aplicação. É o único conjunto que define as classes finais do classificador.
- **BladeSynth:** pré-treino espacial binário normal/defeito com máscaras
  sintéticas. O checkpoint resultante é só uma espinha dorsal candidata: exige
  fine-tuning AeBAD-S e validação antes de poder substituir o modelo ativo.
- **Caixas humanas tipificadas:** após 250 reprovações ou casos complexos, só
  as caixas com tipo explicitamente escolhido pelo inspetor ampliam o candidato
  AeBAD-S; caixas sem tipo não recebem uma classe inventada.

## Datasets que não definem as classes AeBAD-S

| Dataset | Uso permitido | Não fazer |
| --- | --- | --- |
| DLR Aircraft Dent | cabeça binária/localização de amolgadela | chamar `dent` à mesma classe de um defeito AeBAD-S |
| AGDD | localização OBB de defeitos em canopy, preservando pares de vistas | fundir orientação e rótulos com a taxonomia AeBAD-S |
| MVTec AD, KolektorSDD2 | robustez de superfície metálica ou tarefa binária separada | apresentar a métrica como desempenho aeronáutico |
| NEU-DET | pré-treino espacial auxiliar de morfologias localizadas | usar sozinho para decidir normal versus defeituoso |

O gerador `training/rtdetr/scripts/build_training_manifests.py` materializa esta política. Os
candidatos criados a partir dela ficam em `validation_required`; nenhum
checkpoint é promovido automaticamente.

## Execução controlada

Para uma futura sessão de treino, primeiro validar a seleção sem criar um
artefacto:

```powershell
.\.venv\Scripts\python.exe training\rtdetr\scripts\build_training_manifests.py
.\.venv\Scripts\python.exe exploratory_models\resnet\scripts\pretrain_resnet_bladesynth.py --dry-run
```

O pré-treino completo cria apenas uma espinha dorsal candidata; o ajuste
posterior usa AeBAD-S e não substitui o modelo ativo sem a opção explícita de
promoção:

```powershell
.\.venv\Scripts\python.exe exploratory_models\resnet\scripts\pretrain_resnet_bladesynth.py --epochs 5
.\.venv\Scripts\python.exe exploratory_models\resnet\scripts\train_defect_type_classifier.py --backbone-init bladesynth
```
