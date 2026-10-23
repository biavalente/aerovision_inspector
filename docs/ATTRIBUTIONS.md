# Recursos publicos e atribuicoes

Este ficheiro deve acompanhar a demonstracao e qualquer entrega do projeto.
Nenhum recurso listado abaixo e apresentado como dado proprio da equipa.

## Datasets

| Recurso | Uso no prototipo | Ligacao | Nota de utilizacao |
| --- | --- | --- | --- |
| AeBAD | Perfil principal de inspeção visual de pás de motor; treino normal do baseline e demonstração | https://github.com/zhangzilongc/MMR | O classificador experimental usa um split interno de desenvolvimento e não é apresentado como benchmark oficial. Preservar a proveniência por imagem. |
| MMR | Benchmark de referencia para AeBAD e mudanca de dominio | https://github.com/zhangzilongc/MMR | Codigo de investigacao; nao representa certificacao. |
| DLR Aircraft Dent | Perfil futuro para amolgadelas em superficies de aeronaves | https://zenodo.org/records/17900121 | Manter separado do perfil de pas. |
| BladeSynth | Pre-treino ou aumento de diversidade, nunca avaliacao final real | https://doi.org/10.6084/m9.figshare.28658603 | Identificar sempre como sintetico. |

## Bibliotecas e plataformas

| Recurso | Uso | Ligacao |
| --- | --- | --- |
| Python | Backend e treino local | https://www.python.org/ |
| FastAPI | API local | https://fastapi.tiangolo.com/ |
| PyTorch | Modelo de deep learning local | https://pytorch.org/ |
| ResNet-18 ImageNet weights | Inicialização pública do classificador local de tipo em ROI; descarregados uma vez e guardados no computador para treino/inferência offline | https://download.pytorch.org/models/resnet18-f37072fd.pth |
| Ultralytics | Execução local do candidato RT-DETR e avaliação DLR | https://docs.ultralytics.com/models/rtdetr/ |
| YOLOv8n public weights | Inicialização visual genérica do RT-DETR candidato; nenhum rótulo externo de defeito entrou no fine-tuning | https://docs.ultralytics.com/models/yolov8/ |
| OpenCV | Validacao e pre-processamento de imagem | https://opencv.org/ |
| React | Interface local | https://react.dev/ |
| Vite | Ambiente de desenvolvimento da interface | https://vite.dev/ |
| Anomalib | Referência para futura integração PatchCore e benchmarks de anomalia; não é dependência de execução desta versão | https://github.com/open-edge-platform/anomalib |

## Referencias tecnicas e de enquadramento

- EASA AI Roadmap 2.0: https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-roadmap-20
- EASA MLEAP: https://www.easa.europa.eu/en/research-projects/machine-learning-application-approval
- EASA NPA 2025-07: https://www.easa.europa.eu/en/document-library/notices-of-proposed-amendment/npa-2025-07
- Fotometria estereo para defeitos em metais: https://pmc.ncbi.nlm.nih.gov/articles/PMC9655976/

## Declaracao de execucao offline

A aplicacao nao usa APIs externas para inferencia, armazenamento ou tomada de
decisao durante a demonstracao. As ligacoes acima servem apenas para atribuir
datasets, bibliotecas e referencias publicas usados no desenvolvimento.

## Nota de licenciamento

Antes de distribuir, publicar ou usar comercialmente uma versão que inclua
Ultralytics ou pesos associados, a equipa deve verificar e cumprir a licença
aplicável desses recursos. Este protótipo de hackathon é uma demonstração
offline e não uma declaração de licença comercial.

