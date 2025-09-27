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




===========================================


==============================
RESULTADOS SEM "SEM CATEGORIA"

=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.1
CV f1  : 0.4829
Test acc    : 0.7589
Test f1_mac : 0.5439
Test f1_mic : 0.7589



=============================================================
RESULTADOS DA LIMPEZA DE DADOS DE DUPLICATAS POR SUBCATEGORIA
=== Resultados com a limpeza de duplicatas===
[A] Unique por (ID_Sub, Prod_Desc_norm): 12600 linhas (removidas 1423)

=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.25
CV f1  : 0.4634
Test acc    : 0.7359
Test f1_mac : 0.4674
Test f1_mic : 0.7359

============================================

******Isso significa que o modelo piorou?******

Curto e direto: não necessariamente “piorou” — é bem provável que, sem limpar duplicatas, você tinha um pouco de vazamento (mesmo texto indo para treino e teste), o que infla acurácia/F1.
Depois da limpeza, seus números ficaram mais realistas (queda de ~2,3 pp em acc e ~7,6 pp em F1-macro).

=============================================================
RESULTADOS DA LIMPEZA DE DADOS DE DUPLICATAS POR SUBCATEGORIA
=== Resultados com a limpeza de duplicatas===
[B] Unique por (Prod_Desc_norm):        12447 linhas (removidas 1576)

=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.25
CV f1  : 0.4688
Test acc    : 0.7434
Test f1_mac : 0.5039
Test f1_mic : 0.7434

========================================================================
CRIEI DADOS SINTÉTICOS PARA EVITAR SUBCATEGORIAS QUE SÓ TINHAM 1 AMOSTRA

=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.1
CV f1  : 0.4496
Test acc    : 0.7488
Test f1_mac : 0.4647
Test f1_mic : 0.7488

=======================================================
REMOVI AS CLASSES RARAS (TODAS COM MENOS DE 5 AMOSTRAS)

=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.1
CV f1  : 0.5553
Test acc    : 0.7541
Test f1_mac : 0.5790
Test f1_mic : 0.7541


Em uma linha: melhorou — especialmente no que importa para macro-F1.

Leitura rápida dos números

Pós-filtro (<5 removidas): acc 0,7541, F1-macro 0,5790, α=0,1

Melhor pós-dedup anterior: acc 0,7434, F1-macro 0,5039

Original (sem limpeza): acc 0,7589, F1-macro 0,5439

Ganho vs melhor pós-dedup:
+0,0107 em acurácia e +0,0751 em F1-macro (ótimo salto). A acurácia ficou praticamente igual ao original, mas F1-macro superou o original — sinal de que você está classificando melhor as classes restantes, sem ser puxado para baixo por classes minúsculas.

Por que isso aconteceu?

Macro-F1 dá o mesmo peso a cada classe. As classes raras (com ≤4 amostras) costumam ter F1 baixíssimo e derrubam o macro-F1. Ao removê-las, você:

reduz o espaço de rótulos (menos confusão);

melhora a consistência do treinamento/validação;

sobe o macro-F1 de forma legítima para o escopo atual.

O melhor alpha voltou para 0,1 (menos suavização). Com menos ruído/escassez, o NB pode ser um pouco mais “afiado”.

Trade-offs

Cobertura: você deixou de cobrir as classes raras. Em produção, isso pode ser aceitável se houver fallback (“Outros/SEM CATEGORIA”) ou fluxo manual.

Métrica vs realidade: os números refletem o novo escopo. Se seu tráfego real tiver aquelas classes raras, é bom tratá-las à parte (abstenção/roteamento).


==================================================


=======================================================
REMOVI AS CLASSES RARAS (TODAS COM MENOS DE 8 AMOSTRAS)

=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.25
CV f1  : 0.6123
Test acc    : 0.7668
Test f1_mac : 0.6243
Test f1_mic : 0.7668



=======================================================
REMOVI AS CLASSES RARAS (TODAS COM MENOS DE 10 AMOSTRAS)

=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.25
CV f1  : 0.6629
Test acc    : 0.7740
Test f1_mac : 0.6437
Test f1_mic : 0.7740


=== (1) Amostras por subcategoria (ordenado decrescente) ===
[ID_Sub=207] Canos e Conexões Hidráulicas: 1002
[ID_Sub=40] Tomadas, interruptores e plugues: 655
[ID_Sub=5] Cabos Elétricos: 505
[ID_Sub=21] Sanitário: 502
[ID_Sub=317] Artefatos de Concreto: 376
[ID_Sub=38] Proteções Elétricas: 344
[ID_Sub=400] Aduela de Concreto: 326
[ID_Sub=189] Cabos e Fios Elétricos: 290
[ID_Sub=107] Ferragens para Fixar e Montar: 283
[ID_Sub=138] Lâmpadas: 272
[ID_Sub=255] EPI: Equipamentos Proteção Individual: 265
[ID_Sub=124] Ferramentas Manuais: 245
[ID_Sub=203] Tubos e Eletrodutos: 240
[ID_Sub=361] Poste de Concreto: 238
[ID_Sub=236] Acessórios de Portas e Janelas: 208
[ID_Sub=188] Acessórios e Conexões Elétricas: 198
[ID_Sub=139] Luminárias: 158
[ID_Sub=202] Transformadores de Energia: 151
[ID_Sub=280] Verniz e Solvente: 150
[ID_Sub=66] Ar Condicionado: 137
[ID_Sub=287] Outros - Ferragens: 136
[ID_Sub=467] Parafusos e acessórios: 133
[ID_Sub=314] Construção Civil: 129
[ID_Sub=161] Areia, Pedra Brita, Gesso, Cal e Argila: 123
[ID_Sub=398] Tubos e conexões industriais: 118
[ID_Sub=439] Perfis e chapas em aço carbono: 118
[ID_Sub=14] Para Raios e SPDA: 116
[ID_Sub=159] Aços para Construção: 113
[ID_Sub=197] Quadros e Caixas Elétricas: 112
[ID_Sub=169] Cimentos: 109
[ID_Sub=167] Cercados e Segurança: 108
[ID_Sub=183] Telhas: 105
[ID_Sub=163] Barras, Tubos e Chapas Metalon: 103
[ID_Sub=20] Água Fria: 96
[ID_Sub=119] Acessórios e Consumíveis para Ferramentas: 96
[ID_Sub=425] Formas: 95
[ID_Sub=126] Ferramentas para Jardim: 93
[ID_Sub=9] Blocos de Concreto: 90
[ID_Sub=362] Poste Metálico: 84
[ID_Sub=147] Chapas e Painéis: 83
[ID_Sub=300] Automação Industrial: 80
[ID_Sub=165] Tijolos e Blocos Estruturais: 72
[ID_Sub=11] Infraestrutura: 70
[ID_Sub=291] Outros - Materiais de Construção: 69
[ID_Sub=303] Eletrocalhas: 67
[ID_Sub=25] Tubos e conexões para Gases Medicinais: 66
[ID_Sub=125] Ferramentas para Construção: 66
[ID_Sub=214] Registros e Bases: 66
[ID_Sub=293] Outros - Materiais Hidráulicos: 62
[ID_Sub=194] Fontes de Energia: 61
[ID_Sub=267] Adesivos e Colas: 55
[ID_Sub=402] Insumos metálicos: 55
[ID_Sub=166] Calhas, Rufos e Algerosas: 54
[ID_Sub=320] Transportes: 53
[ID_Sub=240] Portas e esquadrias: 52
[ID_Sub=143] Refletores: 51
[ID_Sub=162] Armazenamento, Captação e Tratamento de Água: 51
[ID_Sub=13] Condutos: 50
[ID_Sub=319] Produtos para Limpeza: 49
[ID_Sub=292] Outros - Materiais Elétricos: 48
[ID_Sub=43] Acessórios para Banheiro: 47
[ID_Sub=37] Gesso e Drywall: 47
[ID_Sub=160] Argamassas: 44
[ID_Sub=45] Assentos Sanitários: 44
[ID_Sub=427] Tela, cerca e gabião em aço: 42
[ID_Sub=298] Vidros: 40
[ID_Sub=164] Barreiras de Proteção: 38
[ID_Sub=15] Subestações e Alta Tensão: 38
[ID_Sub=123] Ferramentas Elétricas/Bateria: 35
[ID_Sub=156] Produtos para Madeira: 34
[ID_Sub=19] Água Quente: 33
[ID_Sub=152] Madeiras para Construção: 33
[ID_Sub=122] Ferramentas de Medição: 33
[ID_Sub=204] Acessórios Hidráulicos: 33
[ID_Sub=158] Rodapés: 30
[ID_Sub=474] Canos de Concreto: 30
[ID_Sub=172] Equipamentos para Construção: 29
[ID_Sub=213] Ralos e Grelhas: 29
[ID_Sub=10] Concreto: 29
[ID_Sub=16] Sistemas: 29
[ID_Sub=269] Fitas: 28
[ID_Sub=8] Blocos Cerâmicos: 27
[ID_Sub=313] Conectores: 27
[ID_Sub=302] Baterias: 26
[ID_Sub=217] Torneiras: 26
[ID_Sub=190] Disjuntores e Fusíveis: 26
[ID_Sub=206] Caixas de Esgoto: 25
[ID_Sub=432] maquinas pesadas: 25
[ID_Sub=401] Chapa em alumínio: 24
[ID_Sub=196] Plugs e Adaptadores: 24
[ID_Sub=409] Conteineres: 24
[ID_Sub=27] Equipamentos Elétricos: 23
[ID_Sub=150] Forros: 23
[ID_Sub=371] Acessórios para telhado: 22
[ID_Sub=271] Massa Corrida e Niveladora: 21
[ID_Sub=17] Equipamentos: 21
[ID_Sub=23] Outros: 21
[ID_Sub=399] Barras e tubos INOX: 20
[ID_Sub=294] Outros - Pisos e Revestimentos: 20
[ID_Sub=28] Tubos e conexões: 19
[ID_Sub=436] Manta e Produtos Asfálticos: 19
[ID_Sub=60] Pias e Lavatórios: 19
[ID_Sub=178] Lona, Papelão e Plástico Bolha: 19
[ID_Sub=176] Isolamento: 19
[ID_Sub=195] Nobreaks: 18
[ID_Sub=289] Outros - Iluminação: 18
[ID_Sub=446] Anéis de Vedação: 17
[ID_Sub=24] Equipamentos para Gases Medicinais: 17
[ID_Sub=226] Pisos Cerâmicos: 17
[ID_Sub=106] Fechaduras e Travas: 17
[ID_Sub=346] Usina de Asfalto: 17
[ID_Sub=372] Elevador Cremalheira: 16
[ID_Sub=127] Ferramentas   para Pintura: 16
[ID_Sub=175] Impermeabilizantes: 16
[ID_Sub=229] Porcelanatos: 16
[ID_Sub=262] Sinalização, Letras e Números: 15
[ID_Sub=36] Outros: 15
[ID_Sub=254] Combate a Incêndio: 14
[ID_Sub=104] Escadas Móveis: 14
[ID_Sub=423] Medição e instrumentação: 14
[ID_Sub=250] Câmeras de Segurança e CFTV: 14
[ID_Sub=430] Poste de madeira: 13
[ID_Sub=102] Cintas, Cabos, Cordas, Correntes e Acessórios: 13
[ID_Sub=433] Piso Elevado: 13
[ID_Sub=256] Fechaduras e Travas: 13
[ID_Sub=231] Revestimentos Externos: 13
[ID_Sub=26] Outros: 12
[ID_Sub=129] Ferramentas por Profissão: 12
[ID_Sub=340] Vegetação: 12
[ID_Sub=324] Acessibilidade: 12
[ID_Sub=290] Outros - Madeiras: 12
[ID_Sub=297] Outros - Tintas e Acessórios: 12
[ID_Sub=464] Chapas INOX: 12
[ID_Sub=33] Ferragens: 11
[ID_Sub=322] Serviços Mecânicos: 11
[ID_Sub=365] Placa de obra: 11
[ID_Sub=247] Alarmes e Sensores de Presença: 11
[ID_Sub=232] Revestimentos Internos: 11
[ID_Sub=452] Adesivos e Selantes: 11
[ID_Sub=368] Esportes e Lazer: 11
[ID_Sub=181] Pré-moldados: 10
[ID_Sub=179] Madeiras para Construção: 10
[ID_Sub=227] Pisos Laminados: 10
[ID_Sub=272] Pincéis, Rolos e Acessórios: 10
[ID_Sub=282] Vedação e Impermeabilização: 10
[ID_Sub=333] Controlador de Carga: 10

=== (2) Subcategorias no total ===
146

=== (3) Dados no total ===
11724

=== (4) Amostras repetidas ===
Grupos de descrições repetidas: 0
Linhas repetidas (ocorrências excedentes): 0

Top repetições (até 50 linhas):

[OK] Relatório detalhado de duplicatas salvo em: C:\Users\igord\Documents\PPGC\repos\ppgc_1\trabalho1\cenario\amostras_repetidas_por_desc.csv


=======================================================

=======================================================
REMOVI AS CLASSES RARAS (TODAS COM MENOS DE 50 AMOSTRAS)


=== Melhor configuração (por F1-macro CV) ===
Modelo : ComplementNB
alpha  : 0.25
CV f1  : 0.7741
Test acc    : 0.8159
Test f1_mac : 0.7887
Test f1_mic : 0.8159


=== (2) Subcategorias no total ===
58

=== (3) Dados no total ===
9805



=======================================================
REMOVI AS CLASSES RARAS (TODAS COM MENOS DE 100 AMOSTRAS)

=== Melhor configuração (por F1-macro CV) ===
Modelo : MultinomialNB
alpha  : 0.05
CV f1  : 0.8313
Test acc    : 0.8451
Test f1_mac : 0.8382
Test f1_mic : 0.8451


=== (1) Amostras por subcategoria (ordenado decrescente) ===
[ID_Sub=207] Canos e Conexões Hidráulicas: 1002
[ID_Sub=40] Tomadas, interruptores e plugues: 655
[ID_Sub=5] Cabos Elétricos: 505
[ID_Sub=21] Sanitário: 502
[ID_Sub=317] Artefatos de Concreto: 376
[ID_Sub=38] Proteções Elétricas: 344
[ID_Sub=400] Aduela de Concreto: 326
[ID_Sub=189] Cabos e Fios Elétricos: 290
[ID_Sub=107] Ferragens para Fixar e Montar: 283
[ID_Sub=138] Lâmpadas: 272
[ID_Sub=255] EPI: Equipamentos Proteção Individual: 265
[ID_Sub=124] Ferramentas Manuais: 245
[ID_Sub=203] Tubos e Eletrodutos: 240
[ID_Sub=361] Poste de Concreto: 238
[ID_Sub=236] Acessórios de Portas e Janelas: 208
[ID_Sub=188] Acessórios e Conexões Elétricas: 198
[ID_Sub=139] Luminárias: 158
[ID_Sub=202] Transformadores de Energia: 151
[ID_Sub=280] Verniz e Solvente: 150
[ID_Sub=66] Ar Condicionado: 137
[ID_Sub=287] Outros - Ferragens: 136
[ID_Sub=467] Parafusos e acessórios: 133
[ID_Sub=314] Construção Civil: 129
[ID_Sub=161] Areia, Pedra Brita, Gesso, Cal e Argila: 123
[ID_Sub=398] Tubos e conexões industriais: 118
[ID_Sub=439] Perfis e chapas em aço carbono: 118
[ID_Sub=14] Para Raios e SPDA: 116
[ID_Sub=159] Aços para Construção: 113
[ID_Sub=197] Quadros e Caixas Elétricas: 112
[ID_Sub=169] Cimentos: 109
[ID_Sub=167] Cercados e Segurança: 108
[ID_Sub=183] Telhas: 105
[ID_Sub=163] Barras, Tubos e Chapas Metalon: 103

=== (2) Subcategorias no total ===
33

=== (3) Dados no total ===
8068