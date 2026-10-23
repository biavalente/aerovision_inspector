# AeroVision Hack2026

Protótipo offline para apoiar a revisão humana de imagens de superfícies metálicas aeronáuticas. O sistema recebe imagens, executa um detetor local de amolgadelas, apresenta a evidência espacial e mantém uma trilha auditável para revisão, correção de caixas e decisão humana final.

> Estado do protótipo: demonstração de hackathon e apoio à revisão. Não é um sistema certificado, não determina aeronavegabilidade e não deve aprovar ou reprovar peças autonomamente.

## Começar por aqui

1. Para executar a demonstração, abra [`dashboard/README.md`](dashboard/README.md).
2. Para os resultados do modelo que alimenta a demonstração, abra [`results/rtdetr/README.md`](results/rtdetr/README.md).
3. Para o protocolo de treino RT-DETR, abra [`training/rtdetr/README.md`](training/rtdetr/README.md).
4. Para experiências que foram preservadas mas não promovidas, consulte [`exploratory_models/`](exploratory_models/).
5. Para limites, ética, dados e atribuições, consulte [`docs/`](docs/).

## Mapa do projeto

```text
Aerovision Hack2026/
├── dashboard/                    # Aplicação local: FastAPI + React/Vite
│   ├── backend/app/               # API, persistência, análise e ciclo de revisão
│   ├── frontend/src/              # Interface de inspeção e revisão humana
│   └── models/rtdetr/             # Checkpoint RT-DETR e metadados da demonstração
├── training/rtdetr/               # Scripts e protocolo de treino/avaliação DLR
├── results/rtdetr/                # Métricas, matriz, estudo de erro e 20 painéis
├── exploratory_models/
│   ├── autoencoder/               # Hipótese de normalidade, scripts e resultados
│   └── resnet/                    # Classificador ROI, pré-treino e resultados
├── docs/                          # Arquitetura, ética, dados, atribuições e reprodução
├── data/README.md                 # Política para dados locais não distribuídos
└── .gitignore                     # Protege dados, pesos, bases locais e artefactos runtime
```

## Resultado do modelo de demonstração

O dashboard usa apenas o candidato **RT-DETR continuado em DLR Aircraft Dent**, com limiar operacional congelado na validação DLR (`0,178611`). No teste interno separado de 285 imagens DLR (168 normais, 117 com amolgadela), obteve:

| Medida por imagem | Resultado |
| --- | ---: |
| Accuracy | 88,42% |
| Precisão — `dent` | 84,43% |
| Recall — `dent` | 88,03% |
| F1 — `dent` | 86,19% |
| Especificidade normal | 88,69% |
| AUROC (a partir de scores congelados) | 0,958 |
| AP de anomalia (a partir de scores congelados) | 0,957 |
| mAP@0.50 | 28,03% |
| mAP@0.50:0.95 | 8,46% |
| Inferência CPU média | 0,765 s/imagem |

Estes números pertencem ao corpus público DLR e **não** demonstram desempenho em campo, noutra câmara, iluminação ou geometria de peça. A evidência espacial apresentada no dashboard corresponde às caixas RT-DETR e aos respetivos scores; não é uma prova causal.

## Referências: bibliotecas e datasets públicos

### Bibliotecas e plataformas

| Recurso | Papel no projeto | Referência |
| --- | --- | --- |
| Python | Backend, processamento e scripts de treino | [python.org](https://www.python.org/) |
| FastAPI + Uvicorn | API local e servidor do dashboard | [FastAPI](https://fastapi.tiangolo.com/) · [Uvicorn](https://www.uvicorn.org/) |
| PyTorch | Treino e inferência de deep learning | [pytorch.org](https://pytorch.org/) |
| Ultralytics / RT-DETR | Runtime e treino do detetor de amolgadelas | [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/) |
| OpenCV | Leitura, validação, pré-processamento e evidência visual | [opencv.org](https://opencv.org/) |
| React + Vite | Interface web local | [React](https://react.dev/) · [Vite](https://vite.dev/) |
| SQLite | Histórico de inspeções, revisões e auditoria local | [SQLite](https://www.sqlite.org/) |

### Datasets públicos

| Dataset | Uso concreto no projeto | Estado na demonstração |
| --- | --- | --- |
| [DLR Aircraft Dent](https://zenodo.org/records/17900121) | Treino, validação e teste interno do RT-DETR para `dent`/amolgadela | Modelo ativo de demonstração |
| [AeBAD / MMR](https://github.com/zhangzilongc/MMR) | Investigação de autoencoder de normalidade e ResNet-18 de ROIs | Exploração; não entra na inferência ativa |
| [BladeSynth](https://doi.org/10.6084/m9.figshare.28658603) | Pré-treino espacial sintético do candidato ResNet | Exploração; não entra na inferência ativa |
| [AGDD](https://github.com/core128/AGDD) | Dataset de defeitos em canopy catalogado para futura localização orientada | Não usado nas métricas do modelo ativo |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) e [NEU-DET](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/) | Robustez e tarefas auxiliares, preservando taxonomias próprias | Não usados para alegar desempenho aeronáutico |

Os datasets não são redistribuídos neste repositório. As condições de utilização, proveniência e limitações estão detalhadas em [`docs/ATTRIBUTIONS.md`](docs/ATTRIBUTIONS.md) e [`docs/TRAINING_DATA_PLAN.md`](docs/TRAINING_DATA_PLAN.md).

## Trabalho realizado durante o projeto

- Construção de um dashboard offline com fila de inspeção, upload de imagens, análise local, sobreposição de caixas/evidência, revisão humana interativa, histórico, decisões em lote e auditoria de alterações.
- Definição de uma política em que a análise não pára o fluxo: o sistema regista a evidência e o inspetor toma a decisão final por imagem ou em lote.
- Desenho de dois ciclos de aprendizagem ativa de 250 decisões humanas: um exclusivamente com imagens aprovadas para um candidato de normalidade e outro com peças reprovadas/casos complexos para um candidato supervisionado. Nenhuma promoção é automática.
- Investigação de uma abordagem de anomalia por **autoencoder**: baseline AeBAD-S e candidato treinado com 521 normais AeBAD-S + 1 173 normais DLR. A experiência foi mantida porque testa a hipótese válida de aprender “normalidade” sem depender de uma taxonomia completa de defeitos. Não foi promovida, pois separou mal normais e amolgadelas DLR.
- Investigação de um **ResNet-18 por ROI** para apoiar tipos de defeito AeBAD-S. Também foi experimentado pré-treino espacial sintético BladeSynth; o candidato posterior ficou abaixo do ResNet base na validação interna e não foi promovido.
- Treino e avaliação de um **RT-DETR** supervisionado em DLR Aircraft Dent. Um primeiro ensaio pequeno teve localização insuficiente; uma continuação com toda a partição de treino DLR melhorou significativamente a triagem e a localização, passando a ser o modelo de demonstração sujeito a revisão humana.
- Produção de matrizes de confusão, métricas, tempos de inferência, estudo de erros e exemplos visuais para permitir uma apresentação honesta dos resultados.

## Executar o dashboard

O checkpoint RT-DETR necessário já vem incluído em `dashboard/models/rtdetr/best.pt`. Portanto, após instalar as dependências, o projeto funciona sem download adicional de modelos ou acesso a APIs externas.

1. Instale Python 3.12 ou superior, Node.js e `pnpm`.
2. Abra um terminal na raiz deste repositório e crie o ambiente Python:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r dashboard\backend\requirements.txt
pnpm --dir dashboard\frontend install --frozen-lockfile
```

3. No primeiro terminal, inicie a API local:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --app-dir dashboard --port 8000
```

4. No segundo terminal, inicie a interface:

```powershell
pnpm --dir dashboard\frontend dev
```

5. Abra [http://localhost:5173](http://localhost:5173). O dashboard deve indicar que a inferência é local. Use **Adicionar imagens** para carregar uma ou várias imagens e ver a análise RT-DETR.

As imagens de datasets não são necessárias para executar a aplicação; só são necessárias para repetir treinos ou para a população automática de uma fila de demonstração.

## Dados, modelos e licenças

O repositório inclui apenas o checkpoint RT-DETR de demonstração, com 18,33 MB e hash SHA-256 documentado. Não inclui datasets brutos, imagens de treino, bases SQLite, uploads locais nem outros pesos de terceiros. Os links, as condições de utilização e o papel de cada recurso estão em [`docs/ATTRIBUTIONS.md`](docs/ATTRIBUTIONS.md) e [`data/README.md`](data/README.md).

Antes de redistribuir ou usar operacionalmente o checkpoint, confirme as condições das bibliotecas, pesos de inicialização e dataset descritas em [`dashboard/models/rtdetr/README.md`](dashboard/models/rtdetr/README.md).

## Boas práticas aplicadas

- Separação entre código, dados, pesos, resultados finais e resultados exploratórios.
- Sem inferência cloud, API externa ou transmissão de imagens durante a demonstração.
- Limiar escolhido na validação, antes do teste interno final; resultados e exemplos preservados sem re-treino posterior.
- Proveniência e limites explícitos por dataset; datasets industriais auxiliares não são apresentados como evidência aeronáutica.
- Humano no circuito, histórico de decisões e correção manual de caixas.
- Resultados negativos preservados e explicados, em vez de removidos ou apresentados como modelo ativo.

Consulte [`docs/ETHICS_AND_LIMITATIONS.md`](docs/ETHICS_AND_LIMITATIONS.md) e [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) antes de reutilizar o projeto.

## Licença da entrega

Não foi atribuída uma licença open source a esta submissão sem autorização explícita da equipa. Consulte [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md). As bibliotecas, datasets e pesos de terceiros mantêm as respetivas licenças e condições.
