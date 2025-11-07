# Fluxo Geral da Arquitetura

## 1. Entrada de Dados
**Arquivo:** `materiais_clean.tsv`  
**Local:** `./trabalho2/dados/`

Contém os dados originais dos produtos, organizados em colunas:  
`ID_CAT`, `Cat_Name`, `ID_Sub`, `Sub_Name`, `ID_Prod`, `Prod_Desc`.

Cada linha representa um produto, com sua descrição textual (`Prod_Desc`) e o rótulo verdadeiro (`Sub_Name`, a subcategoria correta).

---

## 2. Pré-processamento e Vetorização (TF-IDF)
**Script:** `trabalho2_pipeline_base.py`  
**Saída:** `tfidf_vectorizer.pkl` e `tfidf_dataset.pkl`  
**Local:** `./trabalho2/tf-idf/`

**Etapas:**
- Leitura e limpeza do arquivo `.tsv` (remoção de espaços e campos vazios).  
- Aplicação do **TF-IDF (Term Frequency – Inverse Document Frequency)** para transformar descrições textuais em vetores numéricos.  
- Armazenamento dos resultados:  
  - `tfidf_vectorizer.pkl`: contém o vocabulário e os pesos IDF (usado em produção).  
  - `tfidf_dataset.pkl`: contém a matriz `X_tfidf` e os rótulos `y`.

**Objetivo:** transformar os textos em uma representação numérica que preserve a relevância dos termos, permitindo que os modelos de aprendizado de máquina interpretem os dados.

---

## 3. Divisão dos Dados (Holdout 80/20)
**Script:** `trabalho2_holdout_split.py`  
**Saída:** `train_test_split.pkl`  
**Local:** `./trabalho2/holdout/`

**Etapas:**
- Carrega o conjunto TF-IDF (`tfidf_dataset.pkl`).  
- Divide os dados em:  
  - 80% para treino (`X_train`, `y_train`)  
  - 20% para teste (`X_test`, `y_test`)  
- Utiliza divisão estratificada, mantendo a proporção de classes.  
- Garante reprodutibilidade (`random_state=42`).

**Objetivo:** criar conjuntos de treino e teste consistentes e equilibrados, usados por todos os modelos de forma padronizada.

---

## 4. Treinamento e Avaliação de Modelos
**Scripts:**
- `modelo_nb.py` → Naïve Bayes  
- `modelo_logreg.py` → Regressão Logística  
- `modelo_svm.py` → Support Vector Machine  
- `modelo_rf.py` → Random Forest  
- `modelo_knn.py` → K-Nearest Neighbors

**Saídas (por modelo):**
- Modelo treinado (`.pkl`)  
- Métricas globais (`.txt`)  
- Relatório por classe (`.tsv`)  
- Matriz de confusão (`.png`)

**Locais:**
- `./trabalho2/modelos/`  
- `./trabalho2/metricas/`  
- `./trabalho2/figuras/`

**Etapas internas (idênticas para todos os modelos):**
1. Carregar dados do holdout.  
2. Criar e configurar o modelo com hiperparâmetros definidos.  
3. Treinar o modelo com `X_train` e `y_train`.  
4. Avaliar no conjunto de teste (`X_test`, `y_test`) usando:  
   - Métrica principal: **F1-macro**  
   - Métricas adicionais: **Acurácia**, **Precision-macro**, **Recall-macro**  
   - Matriz de confusão  
5. Salvar resultados padronizados para comparação.

**Objetivo:** comparar o desempenho de diferentes algoritmos sobre o mesmo conjunto de dados e definir qual apresenta melhor equilíbrio entre precisão e generalização.

---

## 5. Análise e Interpretação
**Saídas:** arquivos `.txt`, `.tsv`, e `.png` nas pastas de métricas e figuras.

**Etapas:**
- Comparação direta das métricas principais entre modelos.  
- Interpretação dos resultados:  
  - Qual modelo teve maior F1-macro;  
  - Quais classes tiveram mais erros (análise da matriz de confusão);  
  - Quais tipos de termos (via TF-IDF) mais influenciaram cada classificação.  
- Identificação de oportunidades de melhoria (exemplo: tuning futuro via cross-validation).
