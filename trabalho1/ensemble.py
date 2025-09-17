# trabalho_pipeline_ensemble.py
# -------------------------------------------------------------------
# Ensemble Learning para o mesmo problema de classificação de subcategorias
# Reaproveita estrutura/artefatos do pipeline Naïve Bayes
# -------------------------------------------------------------------
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# === CONFIG GERAL (iguais ao seu pipeline) ===
RAW_PATH       = Path("./trabalho1/cenario/materiais_clean.tsv")
CLEAN_PATH     = Path("./trabalho1/cenario/materiais_clean_fixed.tsv")
VECTORIZER_PKL = Path("./trabalho1/cenario/tfidf_vectorizer.pkl")
DATASET_PKL    = Path("./trabalho1/cenario/tfidf_dataset.pkl")
SPLIT_PKL      = Path("./trabalho1/cenario/train_test_split.pkl")

# Saídas específicas do ensemble
ENS_MODEL_PKL  = Path("./trabalho1/cenario/ens_model.pkl")
ENS_METRICS_TXT= Path("./trabalho1/cenario/ens_metrics.txt")
ENS_CV_OUT     = Path("./trabalho1/cenario/ens_cv_results.tsv")
ENS_REPORT_CV  = Path("./trabalho1/cenario/ens_report_cv_f1_macro.png")
ENS_REPORT_TE  = Path("./trabalho1/cenario/ens_report_test_performance.png")

AUDIT_CM_PNG   = Path("./trabalho1/cenario/ens_audit_confusion_matrix.png")
AUDIT_PERCLS   = Path("./trabalho1/cenario/ens_audit_per_class.tsv")
AUDIT_ERRORS   = Path("./trabalho1/cenario/ens_audit_errors.tsv")

EXPECTED_COLS = ["ID_CAT","Cat_Name","ID_Sub","Sub_Name","ID_Prod","Prod_Desc"]

# -------------------- Utilitários mínimos de limpeza --------------------
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
    rows=[]
    colnames = EXPECTED_COLS[:]
    start = 0
    if header_present:
        header = _coerce_to_6_fields(_normalize_header_fields(first_split))
        colnames = header; start = 1
    for i in range(start, len(lines)):
        parts = _coerce_to_6_fields([p.strip() for p in lines[i].split("\t")])
        rows.append(parts)
    def canon(name):
        key = name.lower().replace(" ", "").replace("-", "_")
        if key in ("id_cat","idcat"): return "ID_CAT"
        if key in ("cat_name","catname"): return "Cat_Name"
        if key in ("id_sub","idsub"): return "ID_Sub"
        if key in ("sub_name","subname"): return "Sub_Name"
        if key in ("id_prod","idprod"): return "ID_Prod"
        if key in ("prod_desc","proddesc"): return "Prod_Desc"
        return name
    colnames = [canon(c) for c in _normalize_header_fields(colnames)]
    if len(colnames) != 6: colnames = EXPECTED_COLS[:]
    return rows, colnames

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Prod_Desc" not in df.columns:
        raise KeyError(f"Coluna 'Prod_Desc' não encontrada: {list(df.columns)}")
    if "Sub_Name" not in df.columns and "ID_Sub" not in df.columns:
        raise KeyError("Falta o rótulo: 'Sub_Name' e 'ID_Sub' ausentes.")
    return df

def _minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].str.strip()
    has_text = df["Prod_Desc"].astype(str).str.len()>0
    has_label = (df["Sub_Name"].astype(str).str.len()>0) if "Sub_Name" in df.columns \
                else (df["ID_Sub"].astype(str).str.len()>0)
    return df[has_text & has_label].reset_index(drop=True)

# -------------------- Passo 1: limpeza (se precisar) --------------------
def ens_main_1():
    if not RAW_PATH.exists():
        print(f"[Passo 1] PULADO: {RAW_PATH} não existe.")
        return
    print(f"[Passo 1] Lendo bruto: {RAW_PATH}")
    rows, cols = _parse_and_fix_tsv(_read_text_any(RAW_PATH))
    df = pd.DataFrame(rows, columns=cols)
    df = _ensure_required_columns(df)
    df = _minimal_clean(df)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, sep="\t", index=False, encoding="utf-8-sig")
    target_col = "Sub_Name" if "Sub_Name" in df.columns else "ID_Sub"
    print("=== PASSO 1 concluído ===")
    print(f"Saída: {CLEAN_PATH} | Linhas: {len(df):,} | Classes: {df[target_col].nunique():,}")
    for i in range(min(5,len(df))):
        print(f"- [{df[target_col].iloc[i]}] {df['Prod_Desc'].iloc[i]}")

# -------------------- Passo 2: TF-IDF (se precisar) --------------------
def ens_main_2():
    if DATASET_PKL.exists() and VECTORIZER_PKL.exists():
        print("[Passo 2] PULADO: TF-IDF já existe.")
        return
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"{CLEAN_PATH.resolve()} não encontrado (rode ens_main_1).")
    df = pd.read_csv(CLEAN_PATH, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    X_text = df["Prod_Desc"]
    y = df["Sub_Name"] if "Sub_Name" in df.columns else df["ID_Sub"]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", analyzer="word",
        ngram_range=(1,2), min_df=2
    )
    X_tfidf = vectorizer.fit_transform(X_text)
    joblib.dump(vectorizer, VECTORIZER_PKL)
    joblib.dump((X_tfidf, y), DATASET_PKL)
    print("=== PASSO 2 concluído ===")
    print(f"Docs: {X_tfidf.shape[0]:,} | Vocab: {X_tfidf.shape[1]:,} | Classes: {y.nunique():,}")
    print(f"Artefatos: {VECTORIZER_PKL} ; {DATASET_PKL}")

# -------------------- Passo 3: split com regra de raras --------------------
def ens_main_3(test_size: float=0.2, random_state: int=42):
    if not DATASET_PKL.exists():
        raise FileNotFoundError(f"{DATASET_PKL.resolve()} não encontrado (rode ens_main_2).")
    from sklearn.model_selection import train_test_split
    from collections import Counter
    X_tfidf, y = joblib.load(DATASET_PKL)
    y = pd.Series(y).astype(str)
    counts = Counter(y); n_docs = X_tfidf.shape[0]
    rare = [cls for cls,c in counts.items() if c<2]
    ok   = [cls for cls,c in counts.items() if c>=2]
    print("=== PASSO 3: Split ===")
    print(f"Docs: {n_docs:,} | Classes: {len(counts):,} | >=2 amostras: {len(ok)} | raras: {len(rare)}")
    idx_all = np.arange(n_docs)
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
    # inclui raras todas no treino
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
        "classes_total": sorted(counts.keys())
    }, SPLIT_PKL)
    print(f"Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,} -> {SPLIT_PKL}")

# -------------------- Passo 4: Treino Ensemble + CV --------------------
def ens_main_4(cv_splits: int=5, random_state: int=42):
    if not SPLIT_PKL.exists():
        raise FileNotFoundError(f"{SPLIT_PKL.resolve()} não encontrado (rode ens_main_3).")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    split = joblib.load(SPLIT_PKL)
    X_train = split["X_train"]; y_train = split["y_train"]
    X_test  = split["X_test"];  y_test  = split["y_test"]

    print("=== PASSO 4: Ensemble Training ===")
    print(f"Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,} | CV: {cv_splits}x")

    # Modelos base
    mnb  = MultinomialNB(alpha=0.5)
    cnb  = ComplementNB(alpha=0.5)
    # Para muitos rótulos + matriz esparsa grande:
    lr  = LogisticRegression(solver="saga", max_iter=800, n_jobs=-1)  # saga lida bem com sparse e muitas classes
    sgd = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, early_stopping=True, n_iter_no_change=5)
    # LinearSVC não tem probas -> calibrar
    svc = CalibratedClassifierCV(
    estimator=LinearSVC(),
    method="sigmoid",
    cv=3
    )

    base_candidates = [
        ("MultinomialNB", mnb),
        ("ComplementNB",  cnb),
        ("LogReg",        lr),
        ("SGDLog",        sgd),
        ("CalibLinearSVC",svc),
    ]

    # Soft Voting (somente com probas)
    voting_estimators = [
        ("lr", lr), ("sgd", sgd), ("mnb", mnb), ("cnb", cnb), ("svc", svc)
    ]
    vote_soft = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=-1)

    # Stacking (meta LR), usa probas e features originais (passthrough)
    stack = StackingClassifier(
        estimators=[("lr", lr), ("sgd", sgd), ("mnb", mnb), ("cnb", cnb), ("svc", svc)],
        final_estimator=LogisticRegression(max_iter=1000, n_jobs=-1),
        stack_method="predict_proba",
        passthrough=True, n_jobs=-1
    )

    # Avaliação via CV (F1-macro) + teste holdout
    # calcular nº mínimo de amostras por classe no treino
    # CV dinâmica (Solução A já aplicada)
    counts = Counter(y_train)
    min_per_class = min(counts.values())
    safe_splits = max(2, min(cv_splits, min_per_class))
    skf = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=random_state)

    rows = []
    best = None  # (cv_f1, acc_test, name, fitted_model, y_pred, report)

    def eval_and_register(name, model):
        nonlocal best
        print(f">>> Avaliando {name} com {safe_splits}-fold CV...")
        try:
            f1_cv = cross_val_score(
                model, X_train, y_train,
                scoring="f1_macro",
                cv=skf,
                n_jobs=1,              # <---- IMPORTANTE: sem paralelismo externo
                pre_dispatch="1*n_jobs",
                error_score="raise",   # se der erro, aparece
            )
            f1_mean = float(np.mean(f1_cv)); f1_std = float(np.std(f1_cv))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_te = float(accuracy_score(y_test, y_pred))
            f1_te  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            rep    = classification_report(y_test, y_pred, zero_division=0)

            rows.append({
                "model": name,
                "cv_f1_macro_mean": f1_mean,
                "cv_f1_macro_std":  f1_std,
                "test_accuracy":    acc_te,
                "test_f1_macro":    f1_te,
            })
            key = (f1_mean, acc_te)
            if best is None or key > (best[0], best[1]):
                best = (f1_mean, acc_te, name, model, y_pred, rep)
            print(f"[{name}] CV f1_macro={f1_mean:.4f}±{f1_std:.4f} | Test acc={acc_te:.4f} f1={f1_te:.4f}")

        except Exception as e:
            print(f"[{name}] PULADO por erro na CV: {type(e).__name__}: {e}")

    # Avaliar bases
    for name, mdl in base_candidates:
        eval_and_register(name, mdl)
    # Avaliar ensembles
    eval_and_register("VotingSoft", vote_soft)
    eval_and_register("StackingLR", stack)

    # Salvar tabela CV
    cv_df = pd.DataFrame(rows).sort_values(by=["cv_f1_macro_mean","test_accuracy"], ascending=False)
    ENS_CV_OUT.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(ENS_CV_OUT, sep="\t", index=False, encoding="utf-8-sig")

    # Persistir melhor
    best_cv, best_acc, best_name, best_model, y_pred_best, best_report = best
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "cv_f1_macro": best_cv
    }, ENS_MODEL_PKL)

    with ENS_METRICS_TXT.open("w", encoding="utf-8") as f:
        f.write("=== Ensemble (Validation + Test) ===\n")
        f.write(f"Best model : {best_name}\n")
        f.write(f"CV f1_macro: {best_cv:.6f}\n")
        f.write("--- Test ---\n")
        f.write(f"accuracy   : {accuracy_score(y_test, y_pred_best):.6f}\n")
        f.write(f"f1_macro   : {f1_score(y_test, y_pred_best, average='macro', zero_division=0):.6f}\n")
        f.write(f"f1_micro   : {f1_score(y_test, y_pred_best, average='micro', zero_division=0):.6f}\n\n")
        f.write("=== Classification report ===\n")
        f.write(best_report)

    print("\n=== Melhor configuração ===")
    print(f"Modelo : {best_name}")
    print(f"CV f1  : {best_cv:.4f}")
    print(f"Artefatos: {ENS_MODEL_PKL} ; {ENS_METRICS_TXT} ; {ENS_CV_OUT}")

# -------------------- Passo 5: Relatórios visuais --------------------
def ens_main_5():
    if not ENS_CV_OUT.exists():
        raise FileNotFoundError(f"{ENS_CV_OUT.resolve()} não encontrado (rode ens_main_4).")
    import matplotlib.pyplot as plt
    df = pd.read_csv(ENS_CV_OUT, sep="\t", encoding="utf-8-sig")
    # CV F1
    plt.figure(figsize=(9,5))
    for name, g in df.groupby("model"):
        plt.errorbar(
            g.index, g["cv_f1_macro_mean"],
            yerr=g["cv_f1_macro_std"], marker="o", capsize=3, label=name
        )
    plt.xlabel("execuções (por modelo)"); plt.ylabel("CV F1-macro (média ± desvio)")
    plt.title("Validação cruzada - F1-macro"); plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(ENS_REPORT_CV, dpi=150); plt.close()

    # Teste
    fig, ax = plt.subplots(1,2, figsize=(12,5), sharex=True)
    for name, g in df.groupby("model"):
        ax[0].plot(g.index, g["test_accuracy"], marker="o", label=name)
        ax[1].plot(g.index, g["test_f1_macro"], marker="o", label=name)
    for a in ax:
        a.grid(True, alpha=0.3); a.legend(ncol=2, fontsize=8)
    ax[0].set_title("Accuracy no teste"); ax[0].set_ylabel("Accuracy")
    ax[1].set_title("F1-macro no teste"); ax[1].set_ylabel("F1-macro")
    ax[0].set_xlabel("execuções"); ax[1].set_xlabel("execuções")
    plt.tight_layout(); plt.savefig(ENS_REPORT_TE, dpi=150); plt.close()

    print("Gráficos salvos:")
    print(f" - {ENS_REPORT_CV}")
    print(f" - {ENS_REPORT_TE}")

# -------------------- Passo 6: Auditoria (matriz confusão/erros) --------------------
def ens_main_6(max_errors_per_class: int=50):
    if not SPLIT_PKL.exists():
        raise FileNotFoundError("Split ausente (rode ens_main_3).")
    if not ENS_MODEL_PKL.exists():
        raise FileNotFoundError("Modelo ensemble ausente (rode ens_main_4).")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    split = joblib.load(SPLIT_PKL)
    X_test = split["X_test"]; y_test = pd.Series(split["y_test"]).astype(str)
    pack = joblib.load(ENS_MODEL_PKL)
    model = pack["model"]; model_name = pack.get("model_name","Ensemble")

    print("=== Auditoria Ensemble ===")
    print(f"Modelo: {model_name} | Amostras teste: {X_test.shape[0]:,}")

    y_pred = model.predict(X_test)
    # tentar proba
    try:
        proba = model.predict_proba(X_test)
        pred_conf = proba[np.arange(len(y_pred)), np.argmax(proba, axis=1)]
    except Exception:
        pred_conf = np.full(len(y_pred), np.nan, dtype=float)

    labels_sorted = np.unique(np.concatenate([y_test.values, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Matriz de Confusão (teste) - Ensemble")
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(AUDIT_CM_PNG, dpi=150); plt.close()

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
    pd.DataFrame(rows).to_csv(AUDIT_PERCLS, sep="\t", index=False, encoding="utf-8-sig")

    df_err = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred, "pred_confidence": pred_conf})
    df_err = df_err[df_err["y_true"]!=df_err["y_pred"]].copy()
    if max_errors_per_class and max_errors_per_class>0:
        df_err["rank"] = df_err.groupby("y_true")["pred_confidence"].rank(method="first", ascending=False)
        df_err = df_err[df_err["rank"] <= max_errors_per_class].drop(columns=["rank"])
    df_err.to_csv(AUDIT_ERRORS, sep="\t", index=False, encoding="utf-8-sig")

    print("Arquivos gerados:")
    print(f" - Matriz: {AUDIT_CM_PNG}")
    print(f" - Métricas por classe: {AUDIT_PERCLS}")
    print(f" - Erros: {AUDIT_ERRORS}")

# -------------------- Predição (batch e interativa) --------------------
def _load_vectorizer_and_model():
    if not VECTORIZER_PKL.exists(): raise FileNotFoundError("Vectorizer ausente (rode ens_main_2).")
    if not ENS_MODEL_PKL.exists():  raise FileNotFoundError("Modelo ensemble ausente (rode ens_main_4).")
    vectorizer = joblib.load(VECTORIZER_PKL)
    pack = joblib.load(ENS_MODEL_PKL)
    model = pack["model"]; src = pack.get("model_name","Ensemble")
    return model, vectorizer, src

def _build_label_maps():
    if not CLEAN_PATH.exists():
        return {}, {}
    df = pd.read_csv(CLEAN_PATH, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "Sub_Name" in df.columns and "ID_Sub" in df.columns:
        name_to_id = (df[df["Sub_Name"].astype(str).str.len()>0]
                      .groupby("Sub_Name")["ID_Sub"].agg(lambda s: s.value_counts().idxmax()).to_dict())
        id_to_name = (df[df["ID_Sub"].astype(str).str.len()>0]
                      .groupby("ID_Sub")["Sub_Name"].agg(lambda s: s.value_counts().idxmax()).to_dict())
    else:
        name_to_id, id_to_name = {}, {}
    return name_to_id, id_to_name

def ens_predict_descriptions(descriptions, k_top: int=5, proba_warn_threshold: float=0.15):
    model, vectorizer, model_src = _load_vectorizer_and_model()
    name_to_id, id_to_name = _build_label_maps()
    X = vectorizer.transform(descriptions)
    # probas se houver
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        proba = model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        # tentar obter classes do estimador final
        try:
            classes_ = model.classes_
        except Exception:
            # Voting/Stacking: pegar de um estimador base
            classes_ = None
            for est in getattr(model, "estimators_", []):
                if hasattr(est, "classes_"): classes_ = est.classes_; break
        if classes_ is None:
            # fallback: via predict de cada classe (não trivial). assumimos índices coerentes.
            raise RuntimeError("Não foi possível recuperar classes_ para mapear probabilidades.")
        pred_labels = np.array(classes_)[pred_idx]
        pred_scores = proba[np.arange(len(descriptions)), pred_idx]
    else:
        pred_labels = model.predict(X)
        pred_scores = np.ones(len(descriptions), dtype=float)
        # gerar probas uniformes para Top-K
        # tentar inferir n_classes:
        try:
            n_classes = len(model.classes_)
        except Exception:
            n_classes = len(np.unique(pred_labels))
        proba = np.full((len(descriptions), n_classes), 1.0/n_classes, dtype=float)
        classes_ = np.array([str(i) for i in range(n_classes)])

    # Top-K
    k = min(k_top, proba.shape[1])
    topk_idx = np.argsort(-proba, axis=1)[:, :k]
    # classes_
    try:
        cls = np.array(getattr(model, "classes_", classes_))
    except Exception:
        cls = np.array(classes_)

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
        tk_labels = [str(cls[j]) for j in topk_idx[i]]
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
            "model_source": model_src
        })
    return results

def ens_infer_file(input_path: str|None="./trabalho1/cenario/novos_itens.tsv",
                   k_top:int=5, proba_warn_threshold: float=0.15):
    OUT_PRED = Path("./trabalho1/cenario/ens_predictions.tsv")
    if input_path and Path(input_path).exists():
        df_in = pd.read_csv(input_path, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if "Prod_Desc" not in df_in.columns:
            raise KeyError(f"Arquivo {input_path} sem coluna 'Prod_Desc'.")
        texts = df_in["Prod_Desc"].astype(str).tolist()
        print(f"=== INFERÊNCIA (Ensemble) === Lendo {len(texts)} itens de {input_path}")
    else:
        texts = [
            "Disjuntor Bipolar 20A Curva C para trilho DIN",
            "Luminária LED retangular de embutir 4000 K 40 W",
            "Cabo condutor 16mm² EPR BT 1kV",
            "Para-raios de distribuição 12 kV 10 kA",
            "Haste de aterramento 3/4 para SPDA"
        ]
        print("=== INFERÊNCIA (Ensemble) === Usando exemplos embutidos.")
    res = ens_predict_descriptions(texts, k_top=k_top, proba_warn_threshold=proba_warn_threshold)
    pd.DataFrame(res).to_csv(OUT_PRED, sep="\t", index=False, encoding="utf-8-sig")
    print(f"Gerado: {OUT_PRED}")
    for i in range(min(5,len(res))):
        r = res[i]
        print(f"- {r['Prod_Desc']}\n  -> {r['predicted_label']} (p={r['predicted_proba']:.3f}) "
              f"top-{len(r['top_k_labels'])}: {', '.join(r['top_k_labels'])}")

def ens_predict_interactive(k_top:int=5, proba_warn_threshold: float=0.15):
    print("=== Predição interativa (Ensemble) ===")
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
    res = ens_predict_descriptions(buf, k_top=k_top, proba_warn_threshold=proba_warn_threshold)
    print("\nResultados:")
    for r in res:
        print("-"*70)
        print(f"Descrição       : {r['Prod_Desc']}")
        print(f"ID_Sub / Nome   : {r.get('ID_Sub', None)} / {r.get('Sub_Name', None)}")
        warn = " (baixa confiança)" if r['low_confidence_flag'] else ""
        print(f"Pred. (modelo)  : {r['predicted_label']} | p={r['predicted_proba']:.3f}{warn}")
        print(f"Top-{len(r['top_k_labels'])} labels : {', '.join(map(str, r['top_k_labels']))}")
        print(f"Top-{len(r['top_k_probas'])} probas : {', '.join(f'{p:.3f}' for p in r['top_k_probas'])}")
        print(f"Modelo usado    : {r['model_source']}")
    try:
        out_path = Path("./trabalho1/cenario/ens_predictions_interactive.tsv")
        pd.DataFrame(res).to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
        print(f"\nArquivo salvo: {out_path}")
    except Exception:
        pass

# -------------------- Execução encadeada --------------------
if __name__ == "__main__":
    ens_main_1()  # opcional: limpa/saneia se houver RAW_PATH
    ens_main_2()  # garante TF-IDF e dataset
    ens_main_3()  # split com classes raras no treino
    ens_main_4()  # treina bases + Voting + Stacking e escolhe melhor por CV F1-macro
    ens_main_5()  # gráficos (CV e teste)
    ens_main_6()  # auditoria (matriz de confusão, métricas por classe, erros)

    # inferência demo + modo interativo (opcional)
    ens_infer_file()
    a = "s"
    while a == "s":
        ens_predict_interactive(k_top=5, proba_warn_threshold=0.15)
        a = input("Deseja fazer outra predição? (s/n): ").strip().lower()
        if a not in ("s","n"):
            print("Encerrando.")
            break
