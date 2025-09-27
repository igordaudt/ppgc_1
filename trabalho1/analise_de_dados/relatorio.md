| Regime                    | Linhas | Subcats | Modelo (α)               |        Acc |   F1-macro |
| ------------------------- | -----: | ------: | ------------------------ | ---------: | ---------: |
| Original                  |   ~14k |       — | ComplNB (0.1)            |     0.7589 |     0.5439 |
| Dedup A (ID_Sub+desc)     | 12 600 |       — | ComplNB (0.25)           |     0.7359 |     0.4674 |
| Dedup B (desc global)     | 12 447 |       — | ComplNB (0.25)           |     0.7434 |     0.5039 |
| + Sintéticos p/ 1 amostra |      — |       — | ComplNB (0.1)            |     0.7488 |     0.4647 |
| Min ≥5                    |      — |       — | ComplNB (0.1)            |     0.7541 |     0.5790 |
| Min ≥8                    |      — |       — | ComplNB (0.25)           |     0.7668 |     0.6243 |
| Min ≥10                   | 11 724 |     146 | ComplNB (0.25)           |     0.7740 |     0.6437 |
| Min ≥50                   |  9 805 |      58 | ComplNB (0.25)           |     0.8159 |     0.7887 |
| Min ≥100                  |  8 068 |      33 | **MultinomialNB (0.05)** | **0.8451** | **0.8382** |


# Leituras principais

- Deduplicação derrubou as métricas inicialmente → você removeu vazamento (mesmas descrições caindo em treino e teste). Os resultados ficaram mais honestos.

- Remover classes raras (tail) eleva muito o F1-macro: como cada classe pesa igual, classes com 1–4 exemplos puxavam o macro para baixo. Ao subir o corte (≥5, ≥8, ≥10, ≥50, ≥100), você:

- - reduz rótulos (menos confusões),

- - aumenta suporte por classe (menos variância),

- - deixa o problema mais fácil/estável ⇒ cresce acc e F1-macro.

- Mudança de modelo no topo da pirâmide: quando o problema fica mais “bem comportado” (menos classes, mais dados/classe), MultinomialNB com α pequeno superou o ComplementNB — faz sentido, porque o Complement é mais robusto ao desequilíbrio extremo; reduzido o desequilíbrio, o Multinomial tende a ir melhor.

- Conclusão prática

- Não é o mesmo problema quando você sobe o limiar (≥50, ≥100): o ganho é real, mas porque o escopo ficou menor e mais denso. Para produção, você precisa equilibrar desempenho × cobertura de classes.

- O melhor “ponto de operação” costuma ser:

- - um modelo “Head” para as classes com ≥N amostras (onde N≈10–50, escolha data-driven),

- - um fallback para a cauda: “Outros/SEM CATEGORIA”, roteamento manual, ou um classificador de 2º estágio específico para raras.



# Conclusão

Os experimentos comprovaram três pontos-chave:

- Deduplicação expôs métricas mais honestas ao remover vazamento entre treino e teste.

- Classes raras derrubavam fortemente o F1-macro; ao elevar o número mínimo de amostras por subcategoria, o desempenho subiu de forma consistente (até Acc 0,8451 / F1-macro 0,8382 quando o problema ficou mais denso).

- O modelo ótimo depende do regime de dados: com cauda longa e desbalanceamento, o ComplementNB foi mais robusto; reduzido o desequilíbrio, o MultinomialNB com α pequeno superou.

Na prática, porém, precisamos cobrir todas as subcategorias. Logo, a “limpeza” serviu para: (i) validar a eficácia do classificador quando há suporte mínimo por classe e (ii) estabelecer um N_mínimo recomendado de amostras por subcategoria para estabilizar o aprendizado.

Próximos passos (objetivos e ações)
## 1) Dados & Taxonomia

Definir N_mínimo por subcategoria (recomendação inicial: 10–30 amostras reais/representativas).

Mapa de sinônimos e canônicos por domínio (DN↔diâmetro, polegadas↔mm, Ø→“diam”, etc.).

Higienização sistemática: normalização de texto, unificação de unidades, remoção de ruído e duplicatas apenas dentro de cada split (evita vazamento sem perder cobertura).

## 2) Aquisição e enriquecimento de amostras

Mineração interna: históricos do PainelConstru (pedidos, catálogos de fornecedores, e-mails do n8n) para capturar descrições reais.

Rotulagem assistida (active learning): treinar um baseline e priorizar para anotação as instâncias de maior incerteza por subcategoria (equilibra head/tail com baixo custo).

Augmentação controlada (regra/templating de domínio): gerar variações realistas (medidas, tensão, material, formato “1-1/2”, DN200, kV/W/HP, marcas) sem criar combinações impossíveis.

## 3) Modelagem

Features recomendadas: TF-IDF com char n-grams (3–6) + word n-grams (1–2), strip_accents='unicode', sublinear_tf=True, normalizações de domínio.

Classificação hierárquica: ID_CAT → ID_Sub (reduz espaço de decisão e costuma elevar recall nas subclasses).

Head/Tail especializado:

Head (≥ N_mínimo): LinearSVC ou SGD-Logistic/MultinomialNB (comparar).

Tail (< N_mínimo): Nearest-Centroid/Prototypical ou one-vs-rest leve, com top-k sugestão.

Pesos: usar sample_weight (ou class_weight) para refletir a distribuição real e reduzir viés do head.

## 4) Decisões de produção

Abstenção/roteamento: se max_proba < τ, rotular como “Revisar/Outros” e enviar ao fluxo humano.

Top-k no frontend (ex.: top-3) para acelerar conferência humana e melhorar UX.

Critérios de aceite (propostos):

Cobertura: 100% das subcategorias presentes.

Qualidade: F1-macro ≥ 0,65 no conjunto completo; F1-classe ≥ 0,50 para subcats com ≥ N_mínimo; Top-3 ≥ 0,90.

Operação: Abstenção ≤ 10% com fila de revisão.

## 5) Avaliação, monitoração e retraining

Split sem vazamento por Prod_Desc_norm (GroupKFold/GroupShuffle).

Relatórios recorrentes: F1 por classe, pares confundidos, top-k, taxa de abstenção, cobertura por subcat.

Ciclo de melhoria contínua: novas amostras das subcats abaixo do N_mínimo entram em fila de priorização; gatilhos de retrain quando subcat cruza o N_mínimo ou quando há queda significativa de F1/perda de distribuição (drift).

Resultado esperado com o plano

Com todas as subcategorias presentes e N_mínimo atingido por classe, o modelo deve convergir para métricas próximas às vistas no regime “denso”, preservando cobertura total. O pipeline de aquisição dirigida + hierarquia + abstenção garante qualidade em produção hoje, enquanto acelera a chegada do volume necessário para estabilizar a cauda amanhã.