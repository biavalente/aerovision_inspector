# Treino e avaliação RT-DETR

Esta pasta contém o código que gerou e avaliou o candidato de demonstração. O objetivo é deixar separado o pipeline reproduzível de treino dos resultados já congelados em `../../results/rtdetr/`.

## Scripts

| Script | Finalidade |
| --- | --- |
| `build_training_manifests.py` | Cria manifestos auditáveis sem copiar ou fundir datasets incompatíveis. |
| `run_dlr_benchmark.py` | Prepara o protocolo DLR, treina candidatos e produz métricas/artefactos. |
| `continue_rtdetr_full_dlr.py` | Continua o melhor checkpoint de fine-tuning com toda a partição de treino DLR. |
| `evaluate_rtdetr_full_continuation.py` | Escolhe o limiar apenas na validação e avalia uma vez o teste separado. |
| `render_full_dlr_continuation_visuals.py` | Produz os 20 painéis de referência/previsão para apresentação. |
| `calibrate_rtdetr_operating_point.py` | Utilitário de calibração da experiência inicial; preservado por rastreabilidade. |

## Protocolo da continuação selecionada

1. Preparar o split com seed `20261022` e reservar 10% do DLR para teste interno.
2. Usar DLR apenas: treino com 1 556 imagens e validação com 645 imagens.
3. Continuar o checkpoint anterior por 5 épocas, a 384 px, CPU, AdamW e aumentação fraca (`scale 0,15`, `translate 0,03`, `flip 0,5`, sem mosaic/mixup).
4. Escolher o limiar por F1 na validação DLR; não ajustar depois de consultar o teste.
5. Avaliar o teste de 285 imagens e guardar métricas, tempos, matriz, exemplos e estudo de erro.

O JSON compacto do protocolo está em [`configs/recorded_protocol.json`](configs/recorded_protocol.json). O manifesto completo não é distribuído porque contém caminhos locais e referências a imagens de datasets não incluídos.

## Reexecução local

1. Obtenha os datasets com licença válida e siga a estrutura em [`../../data/README.md`](../../data/README.md).
2. Instale as dependências com `pip install -r training/requirements-training.txt`.
3. Na raiz do repositório, comece por validar a proveniência:

```powershell
.\.venv\Scripts\python.exe training\rtdetr\scripts\build_training_manifests.py
```

4. Para executar o benchmark completo de candidatos, use:

```powershell
.\.venv\Scripts\python.exe training\rtdetr\scripts\run_dlr_benchmark.py --output-root dashboard\data\runs
```

5. Após o treino inicial, continue e avalie uma execução específica:

```powershell
.\.venv\Scripts\python.exe training\rtdetr\scripts\continue_rtdetr_full_dlr.py --run-dir dashboard\data\runs\<run_id>
.\.venv\Scripts\python.exe training\rtdetr\scripts\evaluate_rtdetr_full_continuation.py --run-dir dashboard\data\runs\<run_id> --checkpoint dashboard\data\runs\<run_id>\rtdetr\full_dlr_continuation_20261023\weights\best.pt
.\.venv\Scripts\python.exe training\rtdetr\scripts\render_full_dlr_continuation_visuals.py --run-dir dashboard\data\runs\<run_id>
```

Os comandos escrevem pesos e saídas locais em `dashboard/data/`, ignorado por Git. Uma reexecução pode variar com hardware, versões de bibliotecas e pesos de inicialização; não substitua os resultados históricos sem criar uma nova versão e sem repetir a avaliação adequada.

## Limite de interpretação

O RT-DETR resolve somente a tarefa binária `dent` no DLR. Não inferir que reconhece automaticamente fissuras, corrosão, abrasão ou qualquer outro defeito aeronáutico. A inspeção humana, validação externa e critérios de aceitação aprovados continuam obrigatórios.
