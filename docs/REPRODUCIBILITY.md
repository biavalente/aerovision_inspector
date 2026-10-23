# Reprodução e verificação

## O que é reproduzível neste repositório

- Código do dashboard, treino, avaliação e geração de painéis.
- Configuração resumida, seeds, contagens de splits e métricas finais.
- Matriz de confusão, estudo de erro e exemplos visuais do candidato RT-DETR.
- Metadados e resultados de autoencoder e ResNet exploratórios.

## O que foi deliberadamente excluído

- Imagens, anotações e manifests com caminhos locais.
- Pesos de inicialização e modelos de terceiros não usados diretamente na demonstração.
- Base SQLite, uploads, cache, ambientes virtuais e dependências instaladas.

O checkpoint RT-DETR de demonstração (`dashboard/models/rtdetr/best.pt`, 18,33 MB) é a exceção deliberada: está incluído para que os júris possam executar a aplicação sem download adicional. Esta política evita redistribuição acidental de outros materiais licenciados, dados locais e ficheiros pesados. Consulte [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md) para fontes públicas e condições aplicáveis.

## Ambiente

- Python 3.12 recomendado.
- Dependências da aplicação: `dashboard/backend/requirements.txt`.
- Dependências de treino: `training/requirements-training.txt`.
- Frontend: versões fixadas em `dashboard/frontend/pnpm-lock.yaml`.

## Verificação estática realizada nesta entrega

```powershell
.\.venv\Scripts\python.exe -m py_compile <todos os ficheiros Python>
pnpm --dir dashboard\frontend build
```

Os ficheiros fonte foram verificados com compilação Python e o frontend da fonte equivalente foi compilado com sucesso antes da organização final. Para validar localmente depois de instalar dependências, execute novamente os dois comandos com os caminhos desta entrega.

## Como repetir sem contaminar a avaliação

1. Obter os datasets legalmente e preservar as estruturas/labels originais em `datasets/`.
2. Gerar manifestos e verificar os contadores antes de iniciar qualquer treino.
3. Criar uma nova pasta de execução em `dashboard/data/runs/`; nunca substituir os resultados de referência em `results/`.
4. Guardar seed, versão das dependências, hash do checkpoint e regras de split.
5. Escolher hiperparâmetros e limiar apenas na validação. Manter o teste interno isolado até à avaliação final.
6. Produzir matriz, predições, tempos e exemplos a partir de saídas congeladas; se o teste for consultado para alterar parâmetros, criar uma nova validação/teste.
7. Só considerar promoção após teste externo representativo e revisão humana.

## Limite de reprodutibilidade

Mesmo com seed, treino de deep learning pode variar por versão de PyTorch/Ultralytics, CPU/GPU, drivers e pesos de inicialização. A reprodução deve comparar tendências e artefactos, não assumir igualdade binária de métricas. O checkpoint histórico é identificado por SHA-256 em [`../dashboard/models/rtdetr/README.md`](../dashboard/models/rtdetr/README.md).
