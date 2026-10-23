# Arquitetura proposta

```text
Imagem RGB local
      │
      ├─► validação de foco / brilho / exposição
      │
      ├─► RT-DETR local (`dent` / amolgadela)
      │       │
      │       ├─► score operacional e estado da inspeção
      │       ├─► caixas delimitadoras
      │       └─► mapa espacial baseado nas caixas e respetivos scores
      │
      └─► SQLite local
              ├─ resultados do modelo e versão
              ├─ revisões humanas e caixas
              ├─ decisões individuais/em lote
              └─ cadeia de eventos auditável por hash

250 aprovações humanas finais como sem defeito
      │
      └─► candidato do autoencoder primário
              ├─► 80% treino + replay de normais AeBAD-S
              ├─► 20% validação humana, separada por peça
              ├─► recalibração do mapa residual
              └─► validação humana antes de qualquer promoção

250 reprovações humanas ou casos complexos finais
      │
      └─► candidato do classificador secundário de ROI
              ├─► apenas caixas humanas com tipo explícito entram no treino
              ├─► 20% validação humana, separada por peça
              ├─► replay fixo AeBAD-S para preservar as classes existentes
              └─► porta de regressão `view` → validação humana antes de promoção
```

O dashboard não mistura o autoencoder nem o classificador de tipo legado na
evidência ativa. Ambos ficam preservados como candidatos e artefactos de
auditoria: os seus resultados não demonstraram ganho complementar suficiente
para justificar uma fusão visual ou operacional com RT-DETR. Um perfil de
produção teria de medir latência, cobertura, falsos positivos e robustez na
câmara e peça reais.

## Proposta de iluminação multiangular

A solução física proposta para uma futura instalação inclui um anel com oito
quadrantes LED controláveis em torno da câmara. Em vez de uma única imagem, a
estação poderia adquirir uma pequena sequência com pares de quadrantes
opostos, mais uma iluminação difusa. O software guardaria a configuração de
luz em `acquisition_context`, avaliaria a qualidade de cada vista e apresentaria
ao revisor a vista menos afetada por reflexos.

É uma proposta credível de engenharia — iluminação direcional é usada em
fotometria estereoscópica e inspeção de superfícies refletoras — mas **não foi
validada fisicamente neste protótipo offline**. No pitch, deve ser apresentada
como “desenho para a próxima fase de validação”, nunca como uma melhoria já
medida nos datasets públicos. A referência técnica está em
`ATTRIBUTIONS.md`.

## Princípio de não bloqueio

Uma deteção de anomalia ou um caso ambíguo é registado e continua na fila. O
sistema não interrompe automaticamente a linha nem liberta uma peça. O
utilizador pode decidir por imagem ou selecionar várias para aplicar uma decisão
comum; apenas as anotações espaciais permanecem sempre individuais.

## Aprendizagem contínua auditável

Os dois ciclos são independentes e usam contadores separados. Uma aprovação
humana só pode contribuir para o candidato do autoencoder; uma reprovação ou
caso complexo só pode contribuir para o candidato do classificador secundário.
Assim, uma imagem de defeito nunca é ensinada ao modelo de normalidade.

O autoencoder recebe exclusivamente imagens confirmadas normais, replay de
normais já validados, recalibração dos limiares e uma reserva humana de 20%.
O classificador secundário recebe somente regiões humanas com um tipo explícito;
imagens complexas sem tipificação permanecem no lote e no registo auditável,
mas não recebem uma classe fabricada. Os dois candidatos ficam sempre em
`validation_required`; nenhum altera o checkpoint ativo sem validação e ação
humana explícitas.

O autoencoder atual é uma linha de base histórica, não um classificador
aceitável de defeitos: a avaliação AeBAD-S separada revelou AUROC 0,3830. Por
isso, o seu XAI residual não é mostrado no dashboard nem tratado como apoio à
decisão.
