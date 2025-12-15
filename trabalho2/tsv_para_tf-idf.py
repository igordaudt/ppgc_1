# trabalho2_pipeline_base.py
# Pipeline comum: leitura do TSV, limpeza e TF-IDF
# Cada modelo (NB, LogReg, SVM, RF, KNN) faz seu próprio split depois.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ====================== CONFIGURAÇÕES DE CAMINHO ======================

RAW_PATH  = Path("./trabalho2/dados/materiais_clean.tsv")

TFIDF_DIR      = Path("./trabalho2/tf-idf/")
VECTORIZER_PKL = TFIDF_DIR / "tfidf_vectorizer.pkl"
DATASET_PKL    = TFIDF_DIR / "tfidf_dataset.pkl"   # (X_tfidf, y) sem split

REQUIRED_COLS = ["ID_CAT", "Cat_Name", "ID_Sub", "Sub_Name", "ID_Prod", "Prod_Desc"]

# ====================== 1. LEITURA E LIMPEZA BÁSICA ======================

def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """
    Lê o .tsv de materiais e faz uma limpeza mínima:
    - garante que as colunas esperadas existam;
    - remove espaços em branco;
    - descarta linhas sem descrição ou sem rótulo (Sub_Name).
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_path.resolve()}")

    df = pd.read_csv(
        raw_path,
        sep="\t",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False
    )

    # Garante que todas as colunas esperadas existem
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes no TSV: {missing}. Encontradas: {list(df.columns)}")

    # Limpeza simples: strip em colunas string
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Mantém apenas linhas com texto e rótulo
    has_text = df["Prod_Desc"].str.len() > 0
    has_label = df["Sub_Name"].str.len() > 0
    df = df[has_text & has_label].reset_index(drop=True)

    print(f"[BASE] Linhas após limpeza: {len(df):,} | Classes: {df['Sub_Name'].nunique():,}")
    return df

# ====================== 2. GERAÇÃO DO TF-IDF ======================

def build_tfidf(df: pd.DataFrame):
    """
    Gera a matriz TF-IDF a partir de Prod_Desc e salva:
    - o vetorizer em VECTORIZER_PKL
    - o par (X_tfidf, y) em DATASET_PKL
    """
    X_text = df["Prod_Desc"].astype(str)
    y = df["Sub_Name"].astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,          # ignora termos muito raros 
        max_df=0.95        # ignora termos extremamente frequentes
    )

    X_tfidf = vectorizer.fit_transform(X_text)

    TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PKL)
    joblib.dump((X_tfidf, y), DATASET_PKL)

    print(f"[TF-IDF] Docs={X_tfidf.shape[0]:,} | Vocab={X_tfidf.shape[1]:,} | Classes={y.nunique():,}")
    return X_tfidf, y

# ====================== PONTO DE ENTRADA ======================

if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    X_tfidf, y = build_tfidf(df)
    print("[OK] TF-IDF gerado. Use tfidf_dataset.pkl para fazer splits por modelo.")
