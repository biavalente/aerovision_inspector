# Stack tecnológica — AeroVision Inspector

## Resumo

O AeroVision Inspector é um protótipo **local e offline** de apoio à inspeção
visual de componentes metálicos aeronáuticos. A interface web comunica com uma
API local; as imagens, os resultados, o histórico de revisão e os checkpoints
dos modelos permanecem no computador. Não há chamadas a serviços cloud durante
a inferência, armazenamento ou tomada de decisão.

```text
Imagem RGB → controlo de qualidade → modelos locais + XAI → API → SQLite → interface web
```

## Linguagens e tecnologias-base

| Área | Tecnologia utilizada | Finalidade |
| --- | --- | --- |
| Backend e treino | Python 3.12.14 | API, processamento de imagem, inferência e scripts de treino. |
| Frontend | TypeScript 7.0.2, HTML e CSS | Interface web e tipagem do cliente. |
| Dados locais | SQL / SQLite | Inspeções, anotações, decisões, versões de modelo e trilho de auditoria. |
| Runtime frontend | Node.js 24.19.0 e pnpm 11.25.0 | Instalação, desenvolvimento e compilação da interface. |

## Frameworks e ferramentas

| Camada | Ferramenta / framework | Utilização |
| --- | --- | --- |
| Interface | React 19.3.1 e React DOM 19.3.1 | Construção da interface de revisão de imagens. |
| Build do frontend | Vite 8.3.4 e `@vitejs/plugin-react` 6.1.2 | Servidor de desenvolvimento e build da interface. |
| API | FastAPI 0.153.0 | API HTTP local, validação de pedidos, upload de imagens e CORS. |
| Servidor ASGI | Uvicorn 0.54.0 | Execução local da API FastAPI. |
| Deep learning | PyTorch 2.14.1+cpu | Definição, treino e inferência dos modelos. |
| Visão computacional | OpenCV 4.14.0 | Leitura, validação, redimensionamento, pré-processamento e geração de evidência visual. |
| Persistência | SQLite (biblioteca padrão de Python) | Base de dados local; o trilho de auditoria encadeia eventos com SHA-256. |

## Bibliotecas utilizadas

### Backend e IA

| Biblioteca | Versão no ambiente local | Uso no projeto |
| --- | ---: | --- |
| `fastapi` | 0.153.0 | Endpoints e ciclo de vida da API. |
| `uvicorn[standard]` | 0.54.0 | Serviço ASGI local. |
| `pydantic` | 2.14.1 | Esquemas e validação dos dados da API. |
| `python-multipart` | 0.0.32 | Upload de ficheiros via formulário HTTP. |
| `torch` | 2.14.1+cpu | Redes neurais, treino e inferência local em CPU. |
| `opencv-python-headless` | 4.14.0 | Processamento de imagens e mapas de evidência. |
| `numpy` | 2.5.4 | Operações numéricas para imagem e inferência. |
| `Pillow` | 11.3.0 | Dependência de imagem declarada no ambiente Python. |

`numpy` é importado diretamente pelo código, embora não esteja listado de forma
explícita em `dashboard/backend/requirements.txt`; no ambiente atual está presente como
dependência instalada. As restantes bibliotecas Python principais estão
declaradas nesse ficheiro.

### Frontend

Além de React, React DOM e Vite, a interface usa `@types/react` e
`@types/react-dom` (19.3.0) para tipagem. Não foi identificada uma biblioteca
externa de componentes visuais, gestão de estado ou cliente HTTP: o styling é
CSS local e os pedidos são feitos com a API nativa do browser.

## Modelos treinados

| Estado | Modelo / checkpoint | Dados de treino | Papel no sistema |
| --- | --- | --- | --- |
| **Ativo no dashboard de demonstração** | `full_dlr_continuation_20261023` | 1 556 imagens DLR (934 normais, 622 dents); continuação por 5 épocas a 384 px | RT-DETR supervisionado para a classe `dent`/amolgadela. Fornece score por imagem, caixas e mapa espacial baseado em caixas; é apoio à revisão humana. |
| Histórico / candidato | `blade_autoencoder_aebad_s_20261020T142413Z` | 521 imagens normais de AeBAD-S `train/good`; 5 épocas | Autoencoder convolucional de anomalia (entrada 192×192). Não é executado nem apresentado pelo dashboard atual; fica preservado para avaliação e aprendizagem contínua. |
| Legado, não usado na inferência atual | `roi_defect_type_resnet18_20261021T215816Z` | 1 665 recortes ROI AeBAD-S nas condições `background`, `illumination` e `same`; 4 épocas | ResNet-18 de tipos de defeito e Grad-CAM, preservada para auditoria e candidatos de treino. |
| Candidato, não promovido | `bladesynth_resnet18_spatial_pretrain_20261022T053416Z` | 4 250 ROIs sintéticas BladeSynth; validação com 750 ROIs | Pré-treino espacial binário normal/defeito para inicializar uma ResNet-18. Não é usado diretamente pela interface. |
| Candidato, não promovido | `roi_defect_type_resnet18_20261022T054719Z` | Pré-treino BladeSynth seguido de fine-tuning nos mesmos 1 665 ROIs AeBAD-S | Candidato ResNet-18 de tipos de defeito. Ficou em `validation_required`, pois foi inferior ao modelo ativo na validação. |
| Rejeitado / histórico | `blade_classifier_aebad_s_dev_20261020T144442Z` | 640 imagens AeBAD-S de treino e 160 de validação | Classificador antigo de imagem inteira. Teve 22,5% de accuracy de validação e não participa na inferência. |

### Saídas e métricas relevantes

- O autoencoder histórico foi treinado apenas com imagens normais e usa perda L1 de
  reconstrução. No teste AeBAD-S separado, obteve AUROC 0,3830; por isso não é
  tratado como detetor autónomo de defeitos.
- O RT-DETR ativo no dashboard obteve no teste DLR selado F1 por imagem 0,8619,
  precisão 0,8443, recall 0,8803, mAP@0.50 0,2803 e mAP@0.50:0.95 0,0846. Estes
  resultados não demonstram desempenho em peças, câmaras ou condições novas.
- A classe `normal_roi` é uma classe interna de rejeição para reduzir falsos
  alarmes; não é apresentada ao utilizador como um tipo de defeito.

## Dados e explicabilidade

- **DLR Aircraft Dent** é a fonte do RT-DETR que está ativo no dashboard de demonstração.
- **AeBAD-S** sustenta os ramos exploratórios de autoencoder e ResNet ROI.
- **Pesos ImageNet ResNet-18** foram usados apenas para inicializar o classificador
  ROI exploratório antes do treino local.
- **BladeSynth** é sintético e é usado apenas no percurso candidato de
  pré-treino; não valida desempenho operacional.
- A explicabilidade ativa é o mapa espacial RT-DETR baseado nas caixas e
  respetivos scores. Os artefactos históricos do autoencoder e Grad-CAM são
  mantidos apenas para auditoria, não para a decisão ou visualização ativa.

DLR Aircraft Dent, AGDD, MVTec AD, KolektorSDD2 e NEU-DET estão documentados
como recursos de trabalho futuro ou robustez; não definem as classes nem o
modelo ativo desta versão.

## Limites importantes

- O protótipo apoia a revisão humana; não certifica aeronavegabilidade nem
  liberta peças para serviço.
- A inferência e os dados são locais. Não há API de IA externa, base de dados
  cloud ou promoção automática de modelos.
- Anomalib é apenas uma referência para uma futura integração de PatchCore; não
  é uma dependência de execução desta versão.

Para o detalhe de arquitetura, dados e limites de avaliação, consultar
[`ARCHITECTURE.md`](ARCHITECTURE.md), [`MODEL_CARD.md`](MODEL_CARD.md) e
[`TRAINING_DATA_PLAN.md`](TRAINING_DATA_PLAN.md).
