# trabalho_pipeline_random_forest.py
# Random Forest para classificação de subcategorias a partir de TF-IDF
# Reaproveita os mesmos artefatos do pipeline original.

# ---- limite de threads BLAS p/ evitar oversubscription (coloque no topo) ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

from typing import Tuple

# === CONFIG ===
RAW_PATH       = Path("./trabalho1/cenario/materiais_clean.tsv")
CLEAN_PATH     = Path("./trabalho1/cenario/materiais_clean_fixed.tsv")
VECTORIZER_PKL = Path("./trabalho1/cenario/tfidf_vectorizer.pkl")
DATASET_PKL    = Path("./trabalho1/cenario/tfidf_dataset.pkl")
SPLIT_PKL      = Path("./trabalho1/cenario/train_test_split.pkl")

RF_MODEL_PKL   = Path("./trabalho1/cenario/rf_model.pkl")
RF_METRICS_TXT = Path("./trabalho1/cenario/rf_metrics.txt")
RF_CV_OUT      = Path("./trabalho1/cenario/rf_cv_results.tsv")

RF_CM_PNG      = Path("./trabalho1/cenario/rf_confusion_matrix.png")
RF_PERCLS_TSV  = Path("./trabalho1/cenario/rf_per_class.tsv")
RF_ERRORS_TSV  = Path("./trabalho1/cenario/rf_errors.tsv")

EXPECTED_COLS = ["ID_CAT","Cat_Name","ID_Sub","Sub_Name","ID_Prod","Prod_Desc"]

# =================== UTILITÁRIOS BÁSICOS ===================
def _read_text_any(path: Path) -> str:
    for enc in ["utf-8-sig","utf-8","latin-1"]:
        try: return path.read_text(encoding=enc, errors="strict")
        except Exception: pass
    return path.read_text(encoding="utf-8", errors="ignore")

def _normalize_header_fields(fields):
    out=[]
    for f in fields:
        f = re.sub(r"\s+"," ", f.strip().strip('"').strip("'"))
        out.append(f)
    return out

def _coerce_to_6_fields(fields):
    if len(fields) < 6: return fields + [""]*(6-len(fields))
    if len(fields) > 6:
        head, tail = fields[:5], fields[5:]
        return head + ["\t".join(tail)]
    return fields

def _parse_and_fix_tsv(raw: str):
    lines = raw.splitlines()
    if not lines: raise RuntimeError("Arquivo vazio.")
    first_split = lines[0].split("\t")
    header_present = any(h.lower() in "\t".join(first_split).lower()
                         for h in ["prod_desc","sub_name","id_sub"])
    rows=[]; colnames=EXPECTED_COLS[:]; start=0
    if header_present:
        header = _coerce_to_6_fields(_normalize_header_fields(first_split))
        colnames = header; start = 1
    for i in range(start, len(lines)):
        parts = _coerce_to_6_fields([p.strip() for p in lines[i].split("\t")])
        rows.append(parts)

    def canon(name):
        key = name.lower().replace(" ","").replace("-","_")
        if key in ("id_cat","idcat"): return "ID_CAT"
        if key in ("cat_name","catname"): return "Cat_Name"
        if key in ("id_sub","idsub"): return "ID_Sub"
        if key in ("sub_name","subname"): return "Sub_Name"
        if key in ("id_prod","idprod"): return "ID_Prod"
        if key in ("prod_desc","proddesc"): return "Prod_Desc"
        return name

    colnames = [canon(c) for c in _normalize_header_fields(colnames)]
    if len(colnames)!=6: colnames = EXPECTED_COLS[:]
    return rows, colnames

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Prod_Desc" not in df.columns:
        raise KeyError(f"Coluna 'Prod_Desc' não encontrada: {list(df.columns)}")
    if "Sub_Name" not in df.columns and "ID_Sub" not in df.columns:
        raise KeyError("Falta rótulo: 'Sub_Name' e 'ID_Sub' ausentes.")
    return df

def _minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].str.strip()
    has_text = df["Prod_Desc"].astype(str).str.len()>0
    has_label = (df["Sub_Name"].astype(str).str.len()>0) if "Sub_Name" in df.columns \
                else (df["ID_Sub"].astype(str).str.len()>0)
    return df[has_text & has_label].reset_index(drop=True)

# =================== PASSO 1: LIMPEZA (opcional) ===================
def rf_main_1():
    if not RAW_PATH.exists():
        print(f"[RF:Passo 1] PULADO (sem {RAW_PATH.name})."); return
    print(f"[RF:Passo 1] Lendo bruto: {RAW_PATH}")
    rows, cols = _parse_and_fix_tsv(_read_text_any(RAW_PATH))
    df = pd.DataFrame(rows, columns=cols)
    df = _ensure_required_columns(df)
    df = _minimal_clean(df)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, sep="\t", index=False, encoding="utf-8-sig")
    target = "Sub_Name" if "Sub_Name" in df.columns else "ID_Sub"
    print(f"Saída: {CLEAN_PATH} | Linhas: {len(df):,} | Classes: {df[target].nunique():,}")

# =================== PASSO 2: TF-IDF (se precisar) ===================
def rf_main_2():
    if DATASET_PKL.exists() and VECTORIZER_PKL.exists():
        print("[RF:Passo 2] PULADO (TF-IDF já existe)."); return
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"{CLEAN_PATH.resolve()} não encontrado (rode rf_main_1).")
    df = pd.read_csv(CLEAN_PATH, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    X_text = df["Prod_Desc"]
    y = df["Sub_Name"] if "Sub_Name" in df.columns else df["ID_Sub"]

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        analyzer="word", ngram_range=(1,2), min_df=2
    )
    X_tfidf = vectorizer.fit_transform(X_text)
    joblib.dump(vectorizer, VECTORIZER_PKL)
    joblib.dump((X_tfidf, y), DATASET_PKL)
    print(f"[RF:Passo 2] Docs={X_tfidf.shape[0]:,} | Vocab={X_tfidf.shape[1]:,} | Classes={y.nunique():,}")

# =================== PASSO 3: SPLIT (com classes raras ao treino) ===================
def rf_main_3(test_size: float=0.2, random_state: int=42):
    if not DATASET_PKL.exists():
        raise FileNotFoundError(f"{DATASET_PKL.resolve()} não encontrado (rode rf_main_2).")
    from sklearn.model_selection import train_test_split
    from collections import Counter
    X_tfidf, y = joblib.load(DATASET_PKL)
    y = pd.Series(y).astype(str)

    counts = Counter(y)
    rare  = [cls for cls,c in counts.items() if c<2]
    ok    = [cls for cls,c in counts.items() if c>=2]

    idx_all = np.arange(X_tfidf.shape[0])
    mask_ok = y.isin(ok).values
    idx_ok, idx_rare = idx_all[mask_ok], idx_all[~mask_ok]

    if len(idx_ok)>0:
        X_ok = X_tfidf[idx_ok]; y_ok = y.iloc[idx_ok].values
        X_tr_ok, X_te_ok, y_tr_ok, y_te_ok, idx_tr_ok, idx_te_ok = train_test_split(
            X_ok, y_ok, idx_ok, test_size=test_size, random_state=random_state, stratify=y_ok
        )
    else:
        from scipy import sparse
        X_tr_ok = X_te_ok = sparse.csr_matrix((0, X_tfidf.shape[1]))
        y_tr_ok = y_te_ok = np.array([], dtype=str)
        idx_tr_ok = idx_te_ok = np.array([], dtype=int)

    if len(idx_rare)>0:
        from scipy import sparse
        X_train = sparse.vstack([X_tr_ok, X_tfidf[idx_rare]], format="csr")
        y_train = np.concatenate([y_tr_ok, y.iloc[idx_rare].values], axis=0)
        idx_train = np.concatenate([idx_tr_ok, idx_rare], axis=0)
    else:
        X_train, y_train, idx_train = X_tr_ok, y_tr_ok, idx_tr_ok

    X_test, y_test, idx_test = X_te_ok, y_te_ok, idx_te_ok

    joblib.dump({
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "idx_train": idx_train, "idx_test": idx_test,
        "test_size": test_size, "random_state": random_state,
        "classes_total": sorted(counts.keys()),
    }, SPLIT_PKL)

    print(f"[RF:Passo 3] Treino={X_train.shape[0]:,} | Teste={X_test.shape[0]:,} | Raras={len(rare)}")

# =================== PASSO 4: TREINO RF (SVD + RandomForest) ===================
def rf_main_4(cv_splits: int = 2, random_state: int = 42,
              svd_components: int | None = None,
              fast: bool = True,
              use_extratrees: bool = True):
    """
    Treina SVD + (RandomForest ou ExtraTrees) com GridSearch "sem refit" e refit manual leve.
    - fast=True: SVD menor e grade mínima para confirmar funcionamento.
    - use_extratrees=True: troca RF por ExtraTrees (geralmente mais rápido).
    """
    if not SPLIT_PKL.exists():
        raise FileNotFoundError(f"{SPLIT_PKL.resolve()} não encontrado (rode rf_main_3).")

    import time
    from collections import Counter
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    if use_extratrees:
        from sklearn.ensemble import ExtraTreesClassifier as Forest
        forest_name = "ExtraTrees"
    else:
        from sklearn.ensemble import RandomForestClassifier as Forest
        forest_name = "RandomForest"

    split = joblib.load(SPLIT_PKL)
    X_train = split["X_train"]; y_train = split["y_train"]
    X_test  = split["X_test"];  y_test  = split["y_test"]

    # ---- CV segura ----
    counts = Counter(y_train)
    min_per_class = min(counts.values()) if len(counts) > 0 else 2
    safe_splits = max(2, min(cv_splits, min_per_class))
    if fast:
        safe_splits = min(safe_splits, 2)  # 2 dobras p/ garantir fluidez

    skf = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=random_state)

    # ---- SVD size ----
    n_feats = X_train.shape[1]
    if svd_components is None:
        svd_components = 80 if fast else int(min(300, max(50, n_feats - 1)))

    print(f"[{forest_name}:Passo 4] FAST={fast} | SVD comps={svd_components} | CV folds={safe_splits}")

    # ---- Pipeline base (com hiperparâmetros 'default' de busca) ----
    base_forest = Forest(
        n_estimators=120 if fast else 300,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1 if not fast else 2,  # leaf maior acelera
        n_jobs=-1,
        random_state=random_state,
        bootstrap=False if use_extratrees else True,  # ExtraTrees já usa bootstrap=False por padrão
    )

    pipe = Pipeline([
        ("svd", TruncatedSVD(
            n_components=svd_components,
            n_iter=3 if fast else 5,
            random_state=random_state
        )),
        ("rf",  base_forest),
    ])

    # ---- Grade mínima/leve ----
    if fast:
        param_grid = {
            "rf__n_estimators": [120],
            "rf__max_features": ["sqrt"],
            "rf__min_samples_leaf": [2],    # acelera e estabiliza
            "rf__max_depth": [None],
        }
    else:
        param_grid = {
            "rf__n_estimators": [150, 300],
            "rf__max_features": ["sqrt", "log2"],
            "rf__min_samples_leaf": [1, 2],
            "rf__max_depth": [None, 30],
        }

    # ---- GridSearch sem refit (para não re-treinar tudo de novo) ----
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=skf,
        n_jobs=1,          # sem paralelismo externo; a floresta já usa n_jobs=-1
        refit=False,       # <<< ponto crítico: NÃO refaz o treino completo aqui
        verbose=3
    )

    print(f"[{forest_name}] Iniciando GridSearchCV.fit(...) (sem refit)")
    t0 = time.time()
    gscv.fit(X_train, y_train)
    t1 = time.time()
    print(f"[{forest_name}] GridSearchCV concluído em {t1 - t0:.1f}s")
    print(f"Melhor score CV (f1_macro): {gscv.best_score_:.4f}")
    print(f"Melhores params (CV): {gscv.best_params_}")

    # ---- Refit manual e leve (treina UMA vez com best_params) ----
    # Reduzimos um pouco para garantir tempo curto no ambiente atual.
    best_params = dict(gscv.best_params_) if gscv.best_params_ is not None else {}
    # Ex.: force um ajuste leve (se a grade sugerir algo mais caro):
    best_params.setdefault("rf__n_estimators", 120)
    best_params.setdefault("rf__min_samples_leaf", 2)

    # Recria pipeline com os melhores params (e ajustes leves)
    refit_pipe = Pipeline([
        ("svd", TruncatedSVD(
            n_components=svd_components,
            n_iter=3 if fast else 5,
            random_state=random_state
        )),
        ("rf", Forest(
            n_estimators=best_params.get("rf__n_estimators", 120),
            max_depth=best_params.get("rf__max_depth", None),
            max_features=best_params.get("rf__max_features", "sqrt"),
            min_samples_leaf=best_params.get("rf__min_samples_leaf", 2),
            n_jobs=-1,
            random_state=random_state,
            bootstrap=False if use_extratrees else True,
        )),
    ])
    print(f"[{forest_name}] Refit manual: {best_params}")
    t2 = time.time()
    refit_pipe.fit(X_train, y_train)
    t3 = time.time()
    print(f"[{forest_name}] Refit manual concluído em {t3 - t2:.1f}s")

    # ---- Avaliação no teste ----
    y_pred = refit_pipe.predict(X_test)
    acc_te = float(accuracy_score(y_test, y_pred))
    f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_mic = float(f1_score(y_test, y_pred, average="micro", zero_division=0))
    rep    = classification_report(y_test, y_pred, zero_division=0)

    # ---- Salvar artefatos ----
    cv_rows=[]
    for params, mean, std in zip(gscv.cv_results_["params"],
                                 gscv.cv_results_["mean_test_score"],
                                 gscv.cv_results_["std_test_score"]):
        row = {"cv_f1_macro_mean": float(mean), "cv_f1_macro_std": float(std)}
        row.update(params); cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows).sort_values(by=["cv_f1_macro_mean"], ascending=False)
    RF_CV_OUT.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(RF_CV_OUT, sep="\t", index=False, encoding="utf-8-sig")

    joblib.dump({
        "model": refit_pipe,                 # pipeline pronto (svd + forest)
        "best_params": gscv.best_params_,    # da busca (antes do ajuste leve)
        "cv_best_f1_macro": float(gscv.best_score_)
    }, RF_MODEL_PKL)

    with RF_METRICS_TXT.open("w", encoding="utf-8") as f:
        f.write(f"=== {forest_name} (SVD + {forest_name}) ===\n")
        f.write(f"Best params (CV): {gscv.best_params_}\n")
        f.write(f"CV f1_macro     : {gscv.best_score_:.6f}\n")
        f.write("--- Test ---\n")
        f.write(f"accuracy        : {acc_te:.6f}\n")
        f.write(f"f1_macro        : {f1_mac:.6f}\n")
        f.write(f"f1_micro        : {f1_mic:.6f}\n\n")
        f.write("=== Classification report ===\n")
        f.write(rep)

    print(f"[{forest_name}:Passo 4] OK | Test acc={acc_te:.4f} | f1_macro={f1_mac:.4f} | f1_micro={f1_mic:.4f}")
    print(f"Artefatos: {RF_MODEL_PKL} ; {RF_METRICS_TXT} ; {RF_CV_OUT}")


# =================== PASSO 5: AUDITORIA (CM, por classe, erros) ===================
def rf_main_5(max_errors_per_class: int=50):
    if not SPLIT_PKL.exists():  raise FileNotFoundError("Split ausente (rf_main_3).")
    if not RF_MODEL_PKL.exists(): raise FileNotFoundError("Modelo RF ausente (rf_main_4).")

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    split = joblib.load(SPLIT_PKL)
    X_test = split["X_test"]; y_test = pd.Series(split["y_test"]).astype(str)

    pack = joblib.load(RF_MODEL_PKL)
    model = pack["model"]  # pipeline (svd+rf)

    y_pred = model.predict(X_test)
    # probabilidades (RF possui)
    try:
        proba = model.predict_proba(X_test)
        pred_conf = proba[np.arange(len(y_pred)), np.argmax(proba, axis=1)]
    except Exception:
        pred_conf = np.full(len(y_pred), np.nan, dtype=float)

    labels_sorted = np.unique(np.concatenate([y_test.values, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusão (teste) - RF")
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(RF_CM_PNG, dpi=150); plt.close()

    rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    rows=[]
    for label, metrics in rep.items():
        if isinstance(metrics, dict):
            rows.append({
                "label": label,
                "precision": metrics.get("precision", np.nan),
                "recall": metrics.get("recall", np.nan),
                "f1_score": metrics.get("f1-score", np.nan),
                "support": metrics.get("support", np.nan),
            })
    pd.DataFrame(rows).to_csv(RF_PERCLS_TSV, sep="\t", index=False, encoding="utf-8-sig")

    df_err = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred, "pred_confidence": pred_conf})
    df_err = df_err[df_err["y_true"]!=df_err["y_pred"]].copy()
    if max_errors_per_class and max_errors_per_class>0:
        df_err["rank"] = df_err.groupby("y_true")["pred_confidence"].rank(method="first", ascending=False)
        df_err = df_err[df_err["rank"]<=max_errors_per_class].drop(columns=["rank"])
    df_err.to_csv(RF_ERRORS_TSV, sep="\t", index=False, encoding="utf-8-sig")

    print("[RF:Passo 5] Auditoria gerada:")
    print(f" - {RF_CM_PNG}")
    print(f" - {RF_PERCLS_TSV}")
    print(f" - {RF_ERRORS_TSV}")

# =================== PASSO 6: INFERÊNCIA (batch + interativo) ===================
def _load_vectorizer_and_rf() -> Tuple[object, object]:
    if not VECTORIZER_PKL.exists(): raise FileNotFoundError("Vectorizer ausente (rf_main_2).")
    if not RF_MODEL_PKL.exists():  raise FileNotFoundError("Modelo RF ausente (rf_main_4).")
    vectorizer = joblib.load(VECTORIZER_PKL)
    pack = joblib.load(RF_MODEL_PKL)
    model = pack["model"]  # pipeline svd+rf
    return vectorizer, model

def _build_label_maps():
    if not CLEAN_PATH.exists(): return {}, {}
    df = pd.read_csv(CLEAN_PATH, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "Sub_Name" in df.columns and "ID_Sub" in df.columns:
        name_to_id = (df[df["Sub_Name"].astype(str).str.len()>0]
                      .groupby("Sub_Name")["ID_Sub"].agg(lambda s: s.value_counts().idxmax()).to_dict())
        id_to_name = (df[df["ID_Sub"].astype(str).str.len()>0]
                      .groupby("ID_Sub")["Sub_Name"].agg(lambda s: s.value_counts().idxmax()).to_dict())
    else:
        name_to_id, id_to_name = {}, {}
    return name_to_id, id_to_name

def rf_predict_descriptions(descriptions, k_top:int=5, proba_warn_threshold: float=0.15):
    vectorizer, model = _load_vectorizer_and_rf()
    name_to_id, id_to_name = _build_label_maps()

    X = vectorizer.transform(descriptions)   # TF-IDF
    proba = model.predict_proba(X)           # pipeline aplica SVD e RF
    pred_idx = np.argmax(proba, axis=1)

    # tentar recuperar classes
    try:
        classes_ = model.classes_
    except Exception:
        # fallback (não deve ocorrer com RF)
        classes_ = np.array(sorted(set(np.argmax(proba, axis=1))))

    pred_labels = np.array(classes_)[pred_idx]
    pred_scores = proba[np.arange(len(descriptions)), pred_idx]

    k = min(k_top, proba.shape[1])
    topk_idx = np.argsort(-proba, axis=1)[:, :k]

    results=[]
    for i, txt in enumerate(descriptions):
        label = str(pred_labels[i])
        ID_Sub = None; Sub_Name = None
        if label in name_to_id:
            Sub_Name = label; ID_Sub = name_to_id[label]
        elif label in id_to_name:
            ID_Sub = label;  Sub_Name = id_to_name[label]
        elif label.isdigit() and label in id_to_name:
            ID_Sub = label;  Sub_Name = id_to_name[label]

        tk_labels = [str(classes_[j]) for j in topk_idx[i]]
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
            "model_source": "rf_svd",
        })
    return results

def rf_infer_file(input_path: str|None="./trabalho1/cenario/novos_itens.tsv",
                  k_top:int=5, proba_warn_threshold: float=0.15):
    OUT_PRED = Path("./trabalho1/cenario/rf_predictions.tsv")
    if input_path and Path(input_path).exists():
        df_in = pd.read_csv(input_path, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if "Prod_Desc" not in df_in.columns:
            raise KeyError(f"Arquivo {input_path} sem coluna 'Prod_Desc'.")
        texts = df_in["Prod_Desc"].astype(str).tolist()
        print(f"[RF:Infer] Lendo {len(texts)} itens de {input_path}")
    else:
        texts = [
            "Disjuntor Bipolar 20A Curva C para trilho DIN",
            "Luminária LED retangular de embutir 4000 K 40 W",
            "Cabo condutor 16mm² EPR BT 1kV",
            "Para-raios de distribuição 12 kV 10 kA",
            "Haste de aterramento 3/4 para SPDA"
        ]
        print("[RF:Infer] Usando exemplos embutidos.")
    res = rf_predict_descriptions(texts, k_top=k_top, proba_warn_threshold=proba_warn_threshold)
    pd.DataFrame(res).to_csv(OUT_PRED, sep="\t", index=False, encoding="utf-8-sig")
    print(f"Gerado: {OUT_PRED}")
    for i in range(min(5,len(res))):
        r = res[i]
        print(f"- {r['Prod_Desc']}\n  -> {r['predicted_label']} (p={r['predicted_proba']:.3f}) "
              f"top-{len(r['top_k_labels'])}: {', '.join(map(str,r['top_k_labels']))}")

def rf_predict_interactive(k_top:int=5, proba_warn_threshold: float=0.15):
    print("=== Predição interativa (RF) ===")
    print("Digite uma descrição por linha. ENTER vazio para encerrar.")
    buf=[]
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line: break
        buf.append(line)
    if not buf:
        print("Nenhuma descrição informada."); return
    res = rf_predict_descriptions(buf, k_top=k_top, proba_warn_threshold=proba_warn_threshold)
    print("\nResultados:")
    for r in res:
        print("-"*70)
        print(f"Descrição       : {r['Prod_Desc']}")
        print(f"ID_Sub / Nome   : {r.get('ID_Sub', None)} / {r.get('Sub_Name', None)}")
        warn = " (baixa confiança)" if r['low_confidence_flag'] else ""
        print(f"Pred. (modelo)  : {r['predicted_label']} | p={r['predicted_proba']:.3f}{warn}")
        print(f"Top-{len(r['top_k_labels'])} labels : {', '.join(map(str, r['top_k_labels']))}")
        print(f"Top-{len(r['top_k_probas'])} probas : {', '.join(f'{p:.3f}' for p in r['top_k_probas'])}")
        print(f"Modelo usado    : rf_svd")
    try:
        out_path = Path("./trabalho1/cenario/rf_predictions_interactive.tsv")
        pd.DataFrame(res).to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
        print(f"\nArquivo salvo: {out_path}")
    except Exception:
        pass

# =================== EXECUÇÃO ENCADEADA ===================
if __name__ == "__main__":
    rf_main_1()  # opcional
    rf_main_2()  # garante TF-IDF
    rf_main_3()  # split com classes raras
    rf_main_4(cv_splits=5, random_state=42, svd_components=None)  # treino + grid
    rf_main_5()  # auditoria

    # inferência demo + modo interativo (opcional)
    rf_infer_file()
    a = "s"
    while a == "s":
        rf_predict_interactive(k_top=5, proba_warn_threshold=0.15)
        a = input("Deseja fazer outra predição? (s/n): ").strip().lower()
        if a not in ("s","n"):
            print("Encerrando.")
            break
