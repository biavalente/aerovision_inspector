# Dashboard AeroVision

Aplicação local para demonstrar inspeção assistida por IA. O backend FastAPI recebe e guarda as imagens no computador local; o frontend React apresenta a fila, a evidência do detetor, a revisão humana e o histórico. Não há chamadas a APIs externas durante a análise.

## Pré-requisitos

- Python 3.12 recomendado.
- Node.js e `pnpm`.
- O checkpoint RT-DETR validado já está incluído em `models/rtdetr/best.pt`.

## Instalação local

Na raiz do repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r dashboard\backend\requirements.txt
pnpm --dir dashboard\frontend install --frozen-lockfile
```

O checkpoint de demonstração já está em `dashboard/models/rtdetr/best.pt`. Para testar outro candidato local, pode definir a variável `AEROVISION_RTDETR_WEIGHTS` com o respetivo caminho.

## Executar

Em dois terminais, na raiz do repositório:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --app-dir dashboard --reload --port 8000
```

```powershell
pnpm --dir dashboard\frontend dev
```

Abra o endereço apresentado pelo Vite, normalmente `http://localhost:5173`.

## Comportamento importante

- O checkpoint incluído permite inferência imediata; se for removido ou substituído por um caminho inválido, a interface continua acessível e identifica o modelo como indisponível em vez de inventar resultados.
- O resultado `likely_normal`, `likely_anomaly` ou `ambiguous` é uma recomendação de triagem. A decisão final é sempre humana.
- A evidência colorida corresponde a caixas RT-DETR com suavização visual. Não é Grad-CAM e não prova causalidade.
- As imagens carregadas, a base SQLite e os mapas gerados ficam em `dashboard/data/`, que não é versionado.
- A população automática DLR requer os datasets em `../datasets/` e um manifesto produzido pelo benchmark em `dashboard/data/runs/`; em alternativa, podem ser feitas cargas de imagens pelo próprio dashboard.

## Verificação rápida

Depois de instalar dependências, é possível confirmar a compilação do frontend sem iniciar o backend:

```powershell
pnpm --dir dashboard\frontend build
```

Os resultados de referência usados na apresentação encontram-se em [`../results/rtdetr/`](../results/rtdetr/).
