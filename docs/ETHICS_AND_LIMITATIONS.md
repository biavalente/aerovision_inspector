# Ética, segurança e limites de utilização

## Princípios adotados

1. **Humano responsável pela decisão.** O sistema nunca aprova/reprova uma peça por si. A análise é guardada e o inspetor decide após ver imagem, evidência e caixas.
2. **Não interrupção automática.** `likely_anomaly`, `likely_normal` e `ambiguous` são estados de triagem. A operação pode continuar a registar inspeções; cabe ao processo humano decidir ações subsequentes.
3. **Transparência de incerteza.** O dashboard distingue sinal do modelo, qualidade de captura e decisão final. Uma caixa ou mapa não é tratado como prova causal.
4. **Proveniência e finalidade.** Cada dataset conserva a origem e o papel. Dados industriais ou sintéticos não são apresentados como validação aeronáutica.
5. **Revisão antes de promoção.** Aprendizagem ativa produz candidatos versionados, não alterações automáticas ao modelo que atende a demonstração.

## Segurança operacional

- Não usar o protótipo para decisão de aeronavegabilidade, liberação de manutenção, determinação de reparação ou substituição de NDT.
- Validar numa amostra externa representativa da peça, câmara, ótica, iluminação, material e critérios de aceitação reais antes de qualquer aplicação operacional.
- Definir, com o responsável de qualidade, quando repetir uma captura, quando escalar para NDT e como reter evidência de auditoria.
- Tratar deteções e caixas como suspeitas que requerem inspeção, sobretudo perto do limiar e em superfícies especulares.

## Dados e privacidade

- A demonstração é local/offline: não transmite imagens para uma API cloud.
- Imagens carregadas e a base de auditoria são gravadas em `dashboard/data/` no computador local. Essa pasta não é versionada.
- Antes de usar imagens reais, estabelecer retenção, controlo de acesso, anonimização de identificadores e regras de remoção de dados.
- Não incluir imagens, pesos ou bases com restrições de licença no repositório público sem confirmação explícita de direito de redistribuição.

## Limites de explicabilidade

O mapa de evidência mostra onde o detetor propôs caixas e com que score. É útil para orientar a revisão, mas não explica causalmente a decisão do modelo, não valida uma caixa e não substitui a avaliação física da peça.

## Limites de desempenho

Os resultados apresentados são de testes internos sobre datasets públicos. Não há demonstração de robustez suficiente a variações de montagem, escala, reflexos, vibração, sujidade, compressão, desgaste, novas geometrias ou tipos de defeito fora do treino. Métricas agregadas podem esconder falhas importantes em subgrupos de iluminação ou acabamento.

## Referências de enquadramento

As ligações de EASA e outras referências utilizadas como enquadramento estão em [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md). São referências para práticas responsáveis; esta entrega não alega conformidade, aprovação ou certificação EASA.
