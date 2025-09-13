# Passo 1 – Carregar os dados
# trabalho_pipeline.py
from pathlib import Path
import re
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# === CONFIGURAÇÃO DE ARQUIVOS ===
RAW_PATH = Path("./trabalho1/cenario/materiais_clean.tsv")               # entrada bruta (TSV)
CLEAN_PATH = Path("./trabalho1/cenario/materiais_clean_fixed.tsv") # saída do passo 1 (TSV)
VECTORIZER_PKL = Path("./trabalho1/cenario/tfidf_vectorizer.pkl")
DATASET_PKL = Path("./trabalho1/cenario/tfidf_dataset.pkl")

EXPECTED_COLS = ["ID_CAT", "Cat_Name", "ID_Sub", "Sub_Name", "ID_Prod", "Prod_Desc"]

# ============== UTILITÁRIOS PASSO 1 ==============

def _read_text_any(path: Path) -> str:
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception as e:
            last_err = e
    # fallback para não travar por alguns caracteres ruins
    return path.read_text(encoding="utf-8", errors="ignore")

def _normalize_header_fields(fields):
    out = []
    for f in fields:
        f = f.strip().strip('"').strip("'")
        f = re.sub(r"\s+", " ", f)
        out.append(f)
    return out

def _coerce_to_6_fields(fields):
    """
    Garante exatamente 6 colunas:
      - se vier <6, completa com '' até 6
      - se vier >6, junta [5:] dentro da 6ª coluna (Prod_Desc)
    """
    if len(fields) < 6:
        fields = fields + [""] * (6 - len(fields))
    elif len(fields) > 6:
        head = fields[:5]
        tail = fields[5:]
        merged_last = "\t".join(tail)  # preserva conteúdo excedente
        fields = head + [merged_last]
    return fields

def _parse_and_fix_tsv(raw: str):
    lines = raw.splitlines()
    if not lines:
        raise RuntimeError("Arquivo vazio.")

    first_split = lines[0].split("\t")
    header_present = any(h.lower() in "\t".join(first_split).lower() for h in ["prod_desc", "sub_name", "id_sub"])

    rows = []
    colnames = EXPECTED_COLS[:]
    start_idx = 0

    if header_present:
        header = _normalize_header_fields(first_split)
        header = _coerce_to_6_fields(header)
        colnames = header
        start_idx = 1

    for idx in range(start_idx, len(lines)):
        parts = [p.strip() for p in lines[idx].split("\t")]
        parts = _coerce_to_6_fields(parts)
        rows.append(parts)

    # normaliza nomes
    colnames = _normalize_header_fields(colnames)

    # se não havia header, usa os esperados
    if not header_present:
        colnames = EXPECTED_COLS[:]

    # mapear variações comuns
    def canon_map(name):
        key = name.lower().replace(" ", "").replace("-", "_")
        if key in ("id_cat", "idcat"): return "ID_CAT"
        if key in ("cat_name", "catname"): return "Cat_Name"
        if key in ("id_sub", "idsub"): return "ID_Sub"
        if key in ("sub_name", "subname"): return "Sub_Name"
        if key in ("id_prod", "idprod"): return "ID_Prod"
        if key in ("prod_desc", "proddesc"): return "Prod_Desc"
        return name

    colnames = [canon_map(c) for c in colnames]
    if len(colnames) != 6:
        colnames = EXPECTED_COLS[:]

    return rows, colnames

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Prod_Desc" not in df.columns:
        raise KeyError(f"Coluna 'Prod_Desc' não encontrada. Colunas: {list(df.columns)}")
    if "Sub_Name" not in df.columns and "ID_Sub" not in df.columns:
        raise KeyError("Falta o rótulo: não encontrei 'Sub_Name' nem 'ID_Sub'.")
    return df

def _minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    # trim em todas as strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()

    # filtra linhas sem texto/rótulo
    has_text = df["Prod_Desc"].astype(str).str.len() > 0
    if "Sub_Name" in df.columns:
        has_label = df["Sub_Name"].astype(str).str.len() > 0
    else:
        has_label = df["ID_Sub"].astype(str).str.len() > 0

    return df[has_text & has_label].reset_index(drop=True)

# ============== PASSO 1: LIMPEZA/SANEAMENTO ==============

def main_1():
    """
    Lê ./trabalho1/materiais.tsv (bruto), corrige linhas com TABs excedentes/ausentes,
    padroniza para 6 colunas e salva em ./trabalho1/materiais_clean_fixed.tsv (TSV).
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {RAW_PATH.resolve()}")

    print(f"[Passo 1] Lendo bruto: {RAW_PATH}")
    raw = _read_text_any(RAW_PATH)
    rows, colnames = _parse_and_fix_tsv(raw)

    df = pd.DataFrame(rows, columns=colnames)
    df = _ensure_required_columns(df)
    df = _minimal_clean(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, sep="\t", index=False, encoding="utf-8-sig")

    target_col = "Sub_Name" if "Sub_Name" in df.columns else "ID_Sub"
    print("=== PASSO 1 concluído ===")
    print(f"Saída: {CLEAN_PATH}")
    print(f"Linhas válidas: {len(df):,}")
    print(f"Nº de classes: {df[target_col].nunique():,}")
    print(f"Rótulo: {target_col}")
    print("Exemplos:")
    for i in range(min(5, len(df))):
        print(f"- [{df[target_col].iloc[i]}] {df['Prod_Desc'].iloc[i]}")

# ============== PASSO 2: VETORIZAÇÃO TF-IDF ==============

def main_2():
    """
    Lê ./trabalho1/materiais_clean_fixed.tsv, vetoriza Prod_Desc com TF-IDF
    e salva vectorizer e dataset em PKL para os próximos passos.
    """
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo limpo não encontrado: {CLEAN_PATH.resolve()}.\n"
            "Execute primeiro main_1()."
        )

    df = pd.read_csv(CLEAN_PATH, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    X_text = df["Prod_Desc"]
    y = df["Sub_Name"] if "Sub_Name" in df.columns else df["ID_Sub"]

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),  # unigrams e bigrams
        min_df=2             # ignora termos muito raros
    )
    X_tfidf = vectorizer.fit_transform(X_text)

    print("=== PASSO 2 concluído ===")
    print(f"Nº de documentos (produtos): {X_tfidf.shape[0]:,}")
    print(f"Tamanho do vocabulário (features): {X_tfidf.shape[1]:,}")
    print(f"Nº de classes: {y.nunique():,}")

    # salva artefatos
    import joblib
    joblib.dump(vectorizer, VECTORIZER_PKL)
    joblib.dump((X_tfidf, y), DATASET_PKL)
    print("Artefatos salvos:")
    print(f" - {VECTORIZER_PKL}")
    print(f" - {DATASET_PKL}")



# === PASSO 3: DIVISÃO TREINO/TESTE ============================
def main_3(test_size: float = 0.2, random_state: int = 42):
    """
    Carrega (X_tfidf, y) de ./trabalho1/tfidf_dataset.pkl,
    realiza a divisão treino/teste (preferencialmente estratificada)
    e salva artefatos para o Passo 4 (treino do Naïve Bayes).

    - Classes com apenas 1 amostra não podem ser estratificadas; elas são
      encaminhadas integralmente ao TREINO.
    - Demais classes (com >=2 amostras) são divididas com stratify=y.
    """
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from collections import Counter

    if not DATASET_PKL.exists():
        raise FileNotFoundError(
            f"Dataset TF-IDF não encontrado: {DATASET_PKL.resolve()}.\n"
            "Execute main_2() antes."
        )

    # 1) Carregar TF-IDF e rótulos
    X_tfidf, y = joblib.load(DATASET_PKL)  # X: scipy.sparse, y: pandas Series ou array-like
    y = pd.Series(y).astype(str)  # garante série de strings para segurança

    # 2) Diagnóstico de classes
    counts = Counter(y)
    n_docs = X_tfidf.shape[0]
    n_classes = len(counts)
    rare_classes = [cls for cls, cnt in counts.items() if cnt < 2]  # classes com 1 amostra
    ok_classes = [cls for cls, cnt in counts.items() if cnt >= 2]

    print("=== PASSO 3: Split treino/teste ===")
    print(f"Documentos: {n_docs:,}")
    print(f"Classes totais: {n_classes:,}")
    print(f"Classes com >=2 amostras: {len(ok_classes):,}")
    print(f"Classes raras (1 amostra): {len(rare_classes):,}")

    # 3) Índices das amostras por grupo
    idx_all = np.arange(n_docs)
    mask_ok = y.isin(ok_classes).values
    mask_rare = ~mask_ok

    idx_ok = idx_all[mask_ok]
    idx_rare = idx_all[mask_rare]  # irão todos para o TREINO

    # 4) Split estratificado nas classes "ok"
    if len(idx_ok) > 0:
        X_ok = X_tfidf[idx_ok]
        y_ok = y.iloc[idx_ok].values
        X_train_ok, X_test_ok, y_train_ok, y_test_ok, idx_train_ok, idx_test_ok = train_test_split(
            X_ok,
            y_ok,
            idx_ok,  # passamos índices juntos para reconstruir depois
            test_size=test_size,
            random_state=random_state,
            stratify=y_ok
        )
    else:
        # cenário extremo (todas as classes são raras)
        X_train_ok = X_tfidf[:0]
        X_test_ok  = X_tfidf[:0]
        y_train_ok = np.array([], dtype=str)
        y_test_ok  = np.array([], dtype=str)
        idx_train_ok = np.array([], dtype=int)
        idx_test_ok  = np.array([], dtype=int)

    # 5) Incluir as raras totalmente no TREINO
    if len(idx_rare) > 0:
        from scipy import sparse
        X_train = sparse.vstack([X_train_ok, X_tfidf[idx_rare]], format="csr")
        y_train = np.concatenate([y_train_ok, y.iloc[idx_rare].values], axis=0)
        idx_train = np.concatenate([idx_train_ok, idx_rare], axis=0)
    else:
        X_train = X_train_ok
        y_train = y_train_ok
        idx_train = idx_train_ok

    X_test = X_test_ok
    y_test = y_test_ok
    idx_test = idx_test_ok

    # 6) Relatórios
    def summarize_split(name, yy):
        c = Counter(yy)
        total = sum(c.values())
        top5 = c.most_common(5)
        return total, len(c), top5

    n_tr, k_tr, top_tr = summarize_split("train", y_train)
    n_te, k_te, top_te = summarize_split("test", y_test)

    print("\nResumo do split:")
    print(f"Treino: {n_tr:,} amostras | {k_tr:,} classes")
    print(f"Teste : {n_te:,} amostras | {k_te:,} classes")
    print("Top-5 classes no TREINO (classe, contagem):", top_tr)
    print("Top-5 classes no TESTE  (classe, contagem):", top_te)

    # 7) Salvar artefatos do split
    SPLIT_PKL = Path("./trabalho1/cenario/train_test_split.pkl")
    joblib.dump(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "idx_train": idx_train,
            "idx_test": idx_test,
            "test_size": test_size,
            "random_state": random_state,
            "classes_total": sorted(counts.keys()),
        },
        SPLIT_PKL
    )
    print(f"\nArtefatos do split salvos em: {SPLIT_PKL}")

# === PASSO 4: TREINO DO NAÏVE BAYES ============================
def main_4(alpha: float = 1.0):
    """
    Treina modelos Naïve Bayes (MultinomialNB e ComplementNB) com X_train/y_train,
    avalia em X_test/y_test, escolhe o melhor por F1-macro, e salva artefatos.

    Parâmetros:
      - alpha: suavização (Laplace/Lidstone), padrão 1.0

    Saídas:
      - ./trabalho1/nb_model.pkl                 -> melhor modelo treinado
      - ./trabalho1/nb_metrics.txt               -> resumo de métricas no teste
      - ./trabalho1/nb_classification_report.txt -> relatório detalhado por classe
    """
    import joblib
    import numpy as np
    from pathlib import Path
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
    )

    SPLIT_PKL = Path("./trabalho1/cenario/train_test_split.pkl")
    if not SPLIT_PKL.exists():
        raise FileNotFoundError(
            f"Split não encontrado: {SPLIT_PKL.resolve()}.\n"
            "Execute main_3() antes."
        )

    # 1) Carregar split
    split = joblib.load(SPLIT_PKL)
    X_train = split["X_train"]
    y_train = split["y_train"]
    X_test  = split["X_test"]
    y_test  = split["y_test"]

    print("=== PASSO 4: Treino Naïve Bayes ===")
    print(f"Treino: {X_train.shape[0]:,} amostras | Teste: {X_test.shape[0]:,} amostras")

    # 2) Definir candidatos
    candidates = [
        ("MultinomialNB", MultinomialNB(alpha=alpha)),
        ("ComplementNB",  ComplementNB(alpha=alpha)),
    ]

    results = []
    reports = {}

    # 3) Treinar e avaliar cada candidato
    for name, model in candidates:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc     = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)

        results.append({
            "name": name,
            "alpha": alpha,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "model": model
        })

        # relatório completo por classe (pode ser grande; salvaremos em arquivo)
        rep = classification_report(y_test, y_pred, zero_division=0)
        reports[name] = rep

        print(f"\n[{name}]")
        print(f"- accuracy : {acc:.4f}")
        print(f"- f1_macro : {f1_macro:.4f}")
        print(f"- f1_micro : {f1_micro:.4f}")

    # 4) Escolher o melhor por F1-macro
    results_sorted = sorted(results, key=lambda d: (d["f1_macro"], d["accuracy"]), reverse=True)
    best = results_sorted[0]
    best_name = best["name"]
    best_model = best["model"]

    print("\n=== Melhor modelo ===")
    print(f"Modelo : {best_name}")
    print(f"alpha  : {best['alpha']}")
    print(f"acc    : {best['accuracy']:.4f}")
    print(f"f1_mac : {best['f1_macro']:.4f}")
    print(f"f1_mic : {best['f1_micro']:.4f}")

    # 5) Salvar artefatos
    MODELPKL = Path("./trabalho1/cenario/nb_model.pkl")
    METRICS_TXT = Path("./trabalho1/cenario/nb_metrics.txt")
    REPORT_TXT  = Path("./trabalho1/cenario/nb_classification_report.txt")

    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "alpha": best["alpha"],
        "classes_": getattr(best_model, "classes_", None),
    }, MODELPKL)

    # resumo breve
    with METRICS_TXT.open("w", encoding="utf-8") as f:
        f.write("=== Naïve Bayes Evaluation (Test Set) ===\n")
        for r in results_sorted:
            f.write(f"\n[{r['name']}]\n")
            f.write(f"alpha    : {r['alpha']}\n")
            f.write(f"accuracy : {r['accuracy']:.6f}\n")
            f.write(f"f1_macro : {r['f1_macro']:.6f}\n")
            f.write(f"f1_micro : {r['f1_micro']:.6f}\n")
        f.write("\n---\n")
        f.write(f"Best: {best_name} (alpha={best['alpha']})\n")

    # relatório detalhado do melhor
    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"=== Classification report: {best_name} ===\n\n")
        f.write(reports[best_name])

    print("\nArtefatos salvos:")
    print(f" - {MODELPKL}")
    print(f" - {METRICS_TXT}")
    print(f" - {REPORT_TXT}")


# === PASSO 5: INFERÊNCIA / PREVISÃO ============================
def main_5(input_path: str | None = "./trabalho1/cenario/novos_itens.tsv",
           k_top: int = 5,
           proba_warn_threshold: float = 0.15):
    """
    Realiza inferência usando o melhor modelo salvo no Passo 4.
    - Carrega tfidf_vectorizer.pkl e nb_model.pkl.
    - Lê um TSV de entrada (coluna 'Prod_Desc'); se não existir, usa exemplos.
    - Gera arquivo ./trabalho1/predictions.tsv com:
        Prod_Desc, predicted_label, predicted_proba, top_k_labels, top_k_probas, low_confidence_flag

    Parâmetros:
      input_path: caminho de um TSV com coluna 'Prod_Desc' (opcional).
      k_top: quantas classes alternativas retornar (Top-K).
      proba_warn_threshold: probabilidade mínima para não sinalizar baixa confiança.

    Saídas:
      - ./trabalho1/predictions.tsv
    """
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd

    VECTORIZER_PKL = Path("./trabalho1/cenario/tfidf_vectorizer.pkl")
    MODELPKL       = Path("./trabalho1/cenario/nb_model.pkl")
    OUT_PRED       = Path("./trabalho1/cenario/predictions.tsv")

    # 1) Verificações
    if not VECTORIZER_PKL.exists():
        raise FileNotFoundError(f"Vectorizer não encontrado: {VECTORIZER_PKL.resolve()} (execute main_2).")
    if not MODELPKL.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODELPKL.resolve()} (execute main_4).")

    # 2) Carregar artefatos
    vectorizer = joblib.load(VECTORIZER_PKL)
    model_pack = joblib.load(MODELPKL)
    model      = model_pack["model"]
    classes_   = model_pack.get("classes_", getattr(model, "classes_", None))

    # 3) Carregar novos itens (ou usar exemplos)
    df_in = None
    if input_path and Path(input_path).exists():
        df_in = pd.read_csv(input_path, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if "Prod_Desc" not in df_in.columns:
            raise KeyError(f"No arquivo {input_path}, a coluna obrigatória 'Prod_Desc' não foi encontrada. "
                           f"Colunas: {list(df_in.columns)}")
        texts = df_in["Prod_Desc"].astype(str).tolist()
        print(f"=== PASSO 5: Inferência ===\nLendo {len(texts)} itens de: {input_path}")
    else:
        texts = [
            "Disjuntor Bipolar 20A Curva C para trilho DIN",
            "Luminária LED retangular de embutir 4000 K 40 W",
            "Cabo condutor 16mm² EPR BT 1kV",
            "Para-raios de distribuição 12 kV 10 kA",
            "Haste de aterramento 3/4 para SPDA"
        ]
        print("=== PASSO 5: Inferência ===\nNenhum arquivo de entrada encontrado. Usando exemplos embutidos.")

    # 4) Vetorizar com o mesmo vocabulário treinado
    X_new = vectorizer.transform(texts)

    # 5) Predições
    # Alguns modelos Naive Bayes têm predict_proba:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = classes_[pred_idx] if classes_ is not None else model.classes_[pred_idx]
        pred_scores = proba[np.arange(len(texts)), pred_idx]
    else:
        # Fallback: sem probas, usar decision_function se existir (nem sempre nos NB)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_new)
            pred_idx = np.argmax(scores, axis=1)
            pred_labels = classes_[pred_idx] if classes_ is not None else model.classes_[pred_idx]
            # normaliza para pseudo-proba (softmax)
            exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba = exp_s / exp_s.sum(axis=1, keepdims=True)
            pred_scores = proba[np.arange(len(texts)), pred_idx]
        else:
            # Último recurso: só predict, e prob=1.0 dummy
            pred_labels = model.predict(X_new)
            pred_scores = np.ones(len(texts), dtype=float)
            # cria uma "proba" uniforme para top-k
            n_classes = len(classes_) if classes_ is not None else len(model.classes_)
            proba = np.full((len(texts), n_classes), 1.0 / n_classes, dtype=float)

    # 6) Top-K alternativos
    n_classes = proba.shape[1]
    k = min(k_top, n_classes)
    topk_idx = np.argsort(-proba, axis=1)[:, :k]  # índices das k maiores probas
    all_classes = classes_ if classes_ is not None else model.classes_

    topk_labels = [[str(all_classes[j]) for j in row] for row in topk_idx]
    topk_scores = [[float(proba[i, j]) for j in topk_idx[i]] for i in range(len(texts))]

    # 7) Montar DataFrame de saída
    out_rows = []
    for i, txt in enumerate(texts):
        low_conf = float(pred_scores[i]) < float(proba_warn_threshold)
        out_rows.append({
            "Prod_Desc": txt,
            "predicted_label": str(pred_labels[i]),
            "predicted_proba": float(pred_scores[i]),
            "top_k_labels": "|".join(topk_labels[i]),
            "top_k_probas": "|".join(f"{s:.4f}" for s in topk_scores[i]),
            "low_confidence_flag": bool(low_conf)
        })

    df_out = pd.DataFrame(out_rows)
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PRED, sep="\t", index=False, encoding="utf-8-sig")

    print(f"\nGerado arquivo de previsões: {OUT_PRED}")
    print("Prévia (primeiras linhas):")
    for i in range(min(5, len(df_out))):
        print(f"- {df_out.iloc[i]['Prod_Desc']}")
        print(f"  -> {df_out.iloc[i]['predicted_label']} (p={df_out.iloc[i]['predicted_proba']:.3f})")
        print(f"  top-{k} = {df_out.iloc[i]['top_k_labels']}  |  {df_out.iloc[i]['top_k_probas']}")


# === PASSO 6: TUNING (VALIDAÇÃO CRUZADA) ============================
def main_6(
    alphas=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    cv_splits: int = 5,
    random_state: int = 42
):
    """
    Realiza busca em grade de alpha para MultinomialNB e ComplementNB usando
    validação cruzada estratificada (F1-macro). Seleciona o melhor, treina no
    conjunto de treino completo e avalia no teste. Salva artefatos.

    Parâmetros:
      - alphas: tupla/lista de valores de suavização a testar
      - cv_splits: nº de dobras da validação cruzada
      - random_state: semente do gerador para reprodutibilidade

    Saídas:
      - ./trabalho1/nb_tuned_model.pkl
      - ./trabalho1/nb_tuned_metrics.txt
      - ./trabalho1/nb_tuned_cv_results.tsv
    """
    import joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from collections import Counter
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    SPLIT_PKL = Path("./trabalho1/cenario/train_test_split.pkl")
    if not SPLIT_PKL.exists():
        raise FileNotFoundError(
            f"Split não encontrado: {SPLIT_PKL.resolve()}.\n"
            "Execute main_3() antes."
        )

    # 1) Carregar split
    split = joblib.load(SPLIT_PKL)
    X_train = split["X_train"]
    y_train = split["y_train"]
    X_test  = split["X_test"]
    y_test  = split["y_test"]

    print("=== PASSO 6: Tuning (CV) ===")
    print(f"Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
    print(f"Classes (treino): {len(Counter(y_train)):,}")

    # 2) Configurar modelos e grid
    candidates = [
        ("MultinomialNB", MultinomialNB),
        ("ComplementNB",  ComplementNB),
    ]
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rows = []
    best = None  # (f1_macro, acc_test, name, alpha, fitted_model, y_pred, report)

    # 3) Loop de tuning
    for name, cls in candidates:
        for a in alphas:
            model = cls(alpha=a)
            # CV em F1-macro (métrica mais robusta em classes desbalanceadas/muitas classes)
            f1_cv = cross_val_score(
                model, X_train, y_train,
                cv=skf,
                scoring="f1_macro",
                n_jobs=None  # ajuste se quiser paralelizar
            )
            f1_mean = float(np.mean(f1_cv))
            f1_std  = float(np.std(f1_cv))

            # Treina no treino completo e mede rápido no teste
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_te = float(accuracy_score(y_test, y_pred))
            f1_te  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

            rows.append({
                "model": name,
                "alpha": a,
                "cv_f1_macro_mean": f1_mean,
                "cv_f1_macro_std": f1_std,
                "test_accuracy": acc_te,
                "test_f1_macro": f1_te,
            })

            # Seleção pelo melhor F1-macro em CV; em empate, melhor teste acc
            key_curr = (f1_mean, acc_te)
            if best is None or key_curr > (best[0], best[1]):
                rep = classification_report(y_test, y_pred, zero_division=0)
                best = (f1_mean, acc_te, name, a, model, y_pred, rep)

            print(f"[{name} | alpha={a}] CV f1_macro={f1_mean:.4f}±{f1_std:.4f} | "
                  f"Test acc={acc_te:.4f} f1_macro={f1_te:.4f}")

    # 4) Salvar resultados de CV
    cv_df = pd.DataFrame(rows).sort_values(
        by=["cv_f1_macro_mean", "test_accuracy"], ascending=False
    )
    CV_OUT = Path("./trabalho1/cenario/nb_tuned_cv_results.tsv")
    cv_df.to_csv(CV_OUT, sep="\t", index=False, encoding="utf-8-sig")

    # 5) Melhor configuração
    best_f1_cv, best_acc_te, best_name, best_alpha, best_model, y_pred_best, best_report = best
    print("\n=== Melhor configuração (por F1-macro CV) ===")
    print(f"Modelo : {best_name}")
    print(f"alpha  : {best_alpha}")
    print(f"CV f1  : {best_f1_cv:.4f}")
    print(f"Test acc    : {best_acc_te:.4f}")
    print(f"Test f1_mac : {f1_score(y_test, y_pred_best, average='macro', zero_division=0):.4f}")
    print(f"Test f1_mic : {f1_score(y_test, y_pred_best, average='micro', zero_division=0):.4f}")

    # 6) Persistir modelo e relatórios
    MODELPKL = Path("./trabalho1/cenario/nb_tuned_model.pkl")
    METRICS_TXT = Path("./trabalho1/cenario/nb_tuned_metrics.txt")

    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "alpha": best_alpha,
        "classes_": getattr(best_model, "classes_", None),
        "cv_f1_macro": best_f1_cv,
    }, MODELPKL)

    with METRICS_TXT.open("w", encoding="utf-8") as f:
        f.write("=== Naïve Bayes Tuned (Validation + Test) ===\n")
        f.write(f"Best model : {best_name}\n")
        f.write(f"alpha      : {best_alpha}\n")
        f.write(f"CV f1_macro: {best_f1_cv:.6f}\n\n")
        f.write("--- Test set ---\n")
        f.write(f"accuracy   : {accuracy_score(y_test, y_pred_best):.6f}\n")
        f.write(f"f1_macro   : {f1_score(y_test, y_pred_best, average='macro', zero_division=0):.6f}\n")
        f.write(f"f1_micro   : {f1_score(y_test, y_pred_best, average='micro', zero_division=0):.6f}\n\n")
        f.write("=== Classification report ===\n")
        f.write(best_report)

    print("\nArtefatos salvos:")
    print(f" - {MODELPKL}")
    print(f" - {METRICS_TXT}")
    print(f" - {CV_OUT}")


# === PASSO 7: RELATÓRIO VISUAL ============================
def main_7():
    """
    Gera gráficos comparando MultinomialNB e ComplementNB
    em função do alpha, usando resultados salvos no passo 6.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    CV_OUT = Path("./trabalho1/cenario/nb_tuned_cv_results.tsv")
    if not CV_OUT.exists():
        raise FileNotFoundError(
            f"Resultados de tuning não encontrados: {CV_OUT.resolve()}.\n"
            "Execute main_6() antes."
        )

    # 1) Carregar resultados
    df = pd.read_csv(CV_OUT, sep="\t", encoding="utf-8-sig")

    # 2) Plotar CV F1-macro
    plt.figure(figsize=(8, 5))
    for name, g in df.groupby("model"):
        plt.errorbar(
            g["alpha"], g["cv_f1_macro_mean"],
            yerr=g["cv_f1_macro_std"],
            marker="o", capsize=4, label=name
        )
    plt.xscale("log")  # escala log p/ visualizar melhor
    plt.xlabel("alpha (log scale)")
    plt.ylabel("CV F1-macro (média ± desvio)")
    plt.title("Validação cruzada - F1 macro vs alpha")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./trabalho1/cenario/report_cv_f1_macro.png", dpi=150)
    plt.close()

    # 3) Plotar performance no teste
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for name, g in df.groupby("model"):
        ax[0].plot(g["alpha"], g["test_accuracy"], marker="o", label=name)
        ax[1].plot(g["alpha"], g["test_f1_macro"], marker="o", label=name)

    for a in ax:
        a.set_xscale("log")
        a.grid(True, alpha=0.3)
        a.legend()

    ax[0].set_xlabel("alpha (log scale)")
    ax[0].set_ylabel("Test Accuracy")
    ax[0].set_title("Accuracy no teste vs alpha")

    ax[1].set_xlabel("alpha (log scale)")
    ax[1].set_ylabel("Test F1-macro")
    ax[1].set_title("F1-macro no teste vs alpha")

    plt.tight_layout()
    plt.savefig("./trabalho1/cenario/report_test_performance.png", dpi=150)
    plt.close()

    print("=== PASSO 7 concluído ===")
    print("Gráficos salvos em:")
    print(" - ./trabalho1/cenario/report_cv_f1_macro.png")
    print(" - ./trabalho1/cenario/report_test_performance.png")

# === PASSO 8: AUDITORIA - MATRIZ DE CONFUSÃO & ERROS =========================
def main_8(max_errors_per_class: int = 50):
    """
    Gera análise de auditoria do modelo:
      - Matriz de confusão (PNG)
      - Métricas por classe (TSV)
      - Erros (false predictions) com probabilidades (TSV)

    Prioriza o modelo tunado (nb_tuned_model.pkl). Se não existir, usa nb_model.pkl.
    Requer o split salvo em ./trabalho1/train_test_split.pkl (Passo 3).

    Parâmetro:
      - max_errors_per_class: limita quantos erros por classe verdadeira salvar no TSV.
    """
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    # caminhos
    SPLIT_PKL   = Path("./trabalho1/cenario/train_test_split.pkl")
    TUNED_PKL   = Path("./trabalho1/cenario/nb_tuned_model.pkl")
    BASE_PKL    = Path("./trabalho1/cenario/nb_model.pkl")
    VECTORIZER  = Path("./trabalho1/cenario/tfidf_vectorizer.pkl")

    OUT_CM_PNG  = Path("./trabalho1/cenario/audit_confusion_matrix.png")
    OUT_PERCLS  = Path("./trabalho1/cenario/audit_per_class.tsv")
    OUT_ERRORS  = Path("./trabalho1/cenario/audit_errors.tsv")

    # 1) carregar split (X_test/y_test)
    if not SPLIT_PKL.exists():
        raise FileNotFoundError(f"Split não encontrado: {SPLIT_PKL.resolve()} (execute main_3).")
    split = joblib.load(SPLIT_PKL)
    X_test = split["X_test"]
    y_test = pd.Series(split["y_test"]).astype(str)

    # 2) escolher melhor modelo disponível (tuned > baseline)
    model_pack = None
    model_src = None
    if TUNED_PKL.exists():
        model_pack = joblib.load(TUNED_PKL)
        model_src = "tuned"
    elif BASE_PKL.exists():
        model_pack = joblib.load(BASE_PKL)
        model_src = "baseline"
    else:
        raise FileNotFoundError("Nenhum modelo encontrado (nb_tuned_model.pkl ou nb_model.pkl).")

    model   = model_pack["model"]
    classes = model_pack.get("classes_", getattr(model, "classes_", None))
    classes = np.array(classes if classes is not None else getattr(model, "classes_", None))

    print("=== PASSO 8: Auditoria ===")
    print(f"Modelo carregado: {model_src} ({model_pack.get('model_name', type(model).__name__)})")
    print(f"Amostras no teste: {X_test.shape[0]:,} | Nº classes: {len(classes):,}")

    # 3) predições e probabilidades
    y_pred = model.predict(X_test)

    # probabilidades (se suportado)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        pred_idx = np.argmax(proba, axis=1)
        pred_conf = proba[np.arange(proba.shape[0]), pred_idx]
    else:
        pred_conf = np.full(len(y_pred), np.nan, dtype=float)

    # 4) matriz de confusão (não-normalizada + heatmap)
    labels_sorted = np.unique(np.concatenate([y_test.values, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    # plot simples (para não gerar figura gigante, usamos imshow sem anotar cada célula)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Matriz de Confusão (teste)")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT_CM_PNG, dpi=150)
    plt.close()

    # 5) métricas por classe → dataframe
    report_txt = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    # output_dict traz "accuracy", "macro avg", "weighted avg" além das classes
    rows = []
    for label, metrics in report_txt.items():
        if isinstance(metrics, dict):
            rows.append({
                "label": label,
                "precision": metrics.get("precision", np.nan),
                "recall": metrics.get("recall", np.nan),
                "f1_score": metrics.get("f1-score", np.nan),
                "support": metrics.get("support", np.nan),
            })
    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(OUT_PERCLS, sep="\t", index=False, encoding="utf-8-sig")

    # 6) erros: listar falsos com confianças
    df_err = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "pred_confidence": pred_conf
    })
    df_err = df_err[df_err["y_true"] != df_err["y_pred"]].copy()

    # limitar nº de erros por classe verdadeira (para não criar TSV gigante)
    if max_errors_per_class is not None and max_errors_per_class > 0:
        df_err["rank"] = df_err.groupby("y_true")["pred_confidence"]\
                               .rank(method="first", ascending=False)
        df_err = df_err[df_err["rank"] <= max_errors_per_class].drop(columns=["rank"])

    # salvar
    df_err.to_csv(OUT_ERRORS, sep="\t", index=False, encoding="utf-8-sig")

    print("Arquivos gerados:")
    print(f" - Matriz de confusão: {OUT_CM_PNG}")
    print(f" - Métricas por classe: {OUT_PERCLS}")
    print(f" - Erros com confiança: {OUT_ERRORS}")

# === PASSO 9: PREDIÇÃO INTERATIVA VIA PROMPT ================================
def _load_best_model_and_vectorizer():
    """Carrega o melhor modelo disponível (tuned > baseline) e o vectorizer."""


    VECTORIZER_PKL = Path("./trabalho1/cenario/tfidf_vectorizer.pkl") # vetorizar
    TUNED_PKL      = Path("./trabalho1/cenario/nb_tuned_model.pkl") # modelo tunado
    BASE_PKL       = Path("./trabalho1/cenario/nb_model.pkl") # modelo baseline

    if not VECTORIZER_PKL.exists():
        raise FileNotFoundError("Vectorizer não encontrado (execute main_2).")

    vectorizer = joblib.load(VECTORIZER_PKL)

    model_pack = None
    model_src = None
    if TUNED_PKL.exists():
        model_pack = joblib.load(TUNED_PKL)
        model_src = "tuned"
    elif BASE_PKL.exists():
        model_pack = joblib.load(BASE_PKL)
        model_src = "baseline"
    else:
        raise FileNotFoundError("Nenhum modelo encontrado (execute main_4 ou main_6).")

    model   = model_pack["model"]
    classes = model_pack.get("classes_", getattr(model, "classes_", None))

    return model, classes, vectorizer, model_src

def _build_label_maps():
    """
    Cria mapeamentos Sub_Name -> ID_Sub e ID_Sub -> Sub_Name a partir
    do dataset limpo, usando o valor mais frequente quando houver ambiguidade.
    """
    from pathlib import Path
    import pandas as pd

    CLEAN_PATH = Path("./trabalho1/cenario/materiais_clean_fixed.tsv")
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Arquivo limpo não encontrado (execute main_1).")

    df = pd.read_csv(CLEAN_PATH, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)

    # Mapa Sub_Name -> ID_Sub (mais frequente)
    if "Sub_Name" in df.columns and "ID_Sub" in df.columns:
        name_to_id = (
            df[df["Sub_Name"].astype(str).str.len() > 0]
            .groupby("Sub_Name")["ID_Sub"]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )
        # Mapa ID_Sub -> Sub_Name (mais frequente)
        id_to_name = (
            df[df["ID_Sub"].astype(str).str.len() > 0]
            .groupby("ID_Sub")["Sub_Name"]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )
    else:
        # fallback: se só houver um dos campos, retornar mapas vazios
        name_to_id, id_to_name = {}, {}

    return name_to_id, id_to_name

def predict_descriptions(descriptions, k_top: int = 5, proba_warn_threshold: float = 0.15):
    """
    Prediz subcategorias para uma lista de descrições.
    Retorna uma lista de dicts com:
      - Prod_Desc
      - ID_Sub
      - Sub_Name
      - predicted_label (como o modelo enxerga)
      - predicted_proba
      - top_k_labels
      - top_k_probas
      - low_confidence_flag
      - model_source (tuned/baseline)
    """
    

    model, classes, vectorizer, model_src = _load_best_model_and_vectorizer()
    name_to_id, id_to_name = _build_label_maps()

    X = vectorizer.transform(descriptions)

    # Probabilidades, se disponíveis
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        proba = model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        model_classes = classes if classes is not None else getattr(model, "classes_", None)
        pred_labels = np.array(model_classes)[pred_idx]
        pred_scores = proba[np.arange(len(descriptions)), pred_idx]
    else:
        # Fallback: usar predict + prob. dummy
        pred_labels = model.predict(X)
        pred_scores = np.ones(len(descriptions), dtype=float)
        model_classes = classes if classes is not None else getattr(model, "classes_", None)
        # cria "proba" uniforme para top-k
        n_classes = len(model_classes)
        proba = np.full((len(descriptions), n_classes), 1.0 / n_classes, dtype=float)
        pred_idx = np.argmax(proba, axis=1)

    # Top-K
    n_classes = proba.shape[1]
    k = min(k_top, n_classes)
    topk_idx = np.argsort(-proba, axis=1)[:, :k]
    model_classes = np.array(model_classes)

    results = []
    for i, txt in enumerate(descriptions):
        label = str(pred_labels[i])

        # Inferir ID_Sub e Sub_Name a partir do que o modelo entregou
        ID_Sub = None
        Sub_Name = None

        if label in name_to_id:           # modelo classificou por nome
            Sub_Name = label
            ID_Sub = name_to_id.get(label, None)
        elif label in id_to_name:          # modelo classificou por ID
            ID_Sub = label
            Sub_Name = id_to_name.get(label, None)
        else:
            # tentativa adicional: se for número puro, trate como ID
            if label.isdigit() and label in id_to_name:
                ID_Sub = label
                Sub_Name = id_to_name[label]
            else:
                # último recurso: deixa predicted_label e não resolve ID/Nome
                pass

        # Top-K alternativos (mostramos como o modelo enxerga)
        tk_labels = [str(model_classes[j]) for j in topk_idx[i]]
        tk_scores = [float(proba[i, j]) for j in topk_idx[i]]

        results.append({
            "Prod_Desc": txt,
            "ID_Sub": ID_Sub,
            "Sub_Name": Sub_Name,
            "predicted_label": label,
            "predicted_proba": float(pred_scores[i]),
            "top_k_labels": tk_labels,
            "top_k_probas": tk_scores,
            "low_confidence_flag": bool(float(pred_scores[i]) < float(proba_warn_threshold)),
            "model_source": model_src,
        })

    return results

def predict_from_prompt(k_top: int = 5, proba_warn_threshold: float = 0.15):
    """
    Interface simples de prompt:
    - Digite uma descrição por linha (ENTER vazio para encerrar).
    - Mostra ID_Sub, Sub_Name, probabilidade e Top-K.
    """
    print("=== Predição interativa ===")
    print("Digite uma descrição por linha. Pressione ENTER vazio para finalizar.")
    buf = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        buf.append(line)

    if not buf:
        print("Nenhuma descrição informada.")
        return

    results = predict_descriptions(buf, k_top=k_top, proba_warn_threshold=proba_warn_threshold)

    print("\nResultados:")
    for r in results:
        print("-" * 70)
        print(f"Descrição       : {r['Prod_Desc']}")
        print(f"ID_Sub / Nome   : {r.get('ID_Sub', None)} / {r.get('Sub_Name', None)}")
        print(f"Pred. (modelo)  : {r['predicted_label']}  |  p={r['predicted_proba']:.3f}  "
              f"{'(baixa confiança)' if r['low_confidence_flag'] else ''}")
        print(f"Top-{len(r['top_k_labels'])} labels : {', '.join(r['top_k_labels'])}")
        print(f"Top-{len(r['top_k_probas'])} probas : {', '.join(f'{p:.3f}' for p in r['top_k_probas'])}")
        print(f"Modelo usado    : {r['model_source']}")

    # (Opcional) salvar TSV com as previsões interativas
    try:
        import pandas as pd
        out_path = Path("./trabalho1/cenario/predictions_interactive.tsv")
        pd.DataFrame(results).to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
        print(f"\nArquivo salvo: {out_path}")
    except Exception:
        pass

def main_9():
    """Entry point para a predição interativa."""
    predict_from_prompt(k_top=5, proba_warn_threshold=0.15)



# ============== EXECUÇÃO ENCADEADA ==============

if __name__ == "__main__":
    main_1()  # limpeza/saneamento
    main_2()  # TF-IDF
    main_3()  # split treino/teste
    main_4()  # treino Naïve Bayes (auto-escolha entre MultinomialNB e ComplementNB)
    main_5()  # inferência em novos itens (exemplos embutidos ou arquivo ./trabalho1/novos_itens.tsv)
    main_6()  # tuning com validação cruzada
    main_7()  # relatório visual
    main_8()  # auditoria (matriz de confusão, métricas por classe, erros)
    a="s"
    while a=="s":
        main_9()  # predição interativa via prompt
        a=input("Deseja fazer outra predição? (s/n): ")
        a=a.lower()
        if a!="s" and a!="n":
            print("Encerrando.")
            break