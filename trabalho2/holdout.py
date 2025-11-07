# trabalho2_holdout_split.py
# Divisão dos dados (holdout 80/20) para treino e teste

import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# ===================== CONFIGURAÇÕES =====================
DATASET_PKL = Path("./trabalho2/tf-idf/tfidf_dataset.pkl")
HOLDOUT_DIR = Path("./trabalho2/holdout/")
SPLIT_PKL   = HOLDOUT_DIR / "train_test_split.pkl"

# ===================== 1. Carregar dataset TF-IDF =====================
print("[HOLDOUT] Carregando tfidf_dataset.pkl ...")
X_tfidf, y = joblib.load(DATASET_PKL)

print(f"[HOLDOUT] Total de amostras: {X_tfidf.shape[0]:,} | "
      f"Dimensões TF-IDF: {X_tfidf.shape[1]:,} | "
      f"Classes: {len(set(y)):,}")

# ===================== 2. Dividir em treino/teste =====================
# Estratégia: 80% treino, 20% teste, estratificado por classe
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"[HOLDOUT] Divisão concluída:")
print(f"  Treino: {X_train.shape[0]:,} amostras")
print(f"  Teste : {X_test.shape[0]:,} amostras")

# ===================== 3. Salvar resultados =====================
HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

split_data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test":  X_test,
    "y_test":  y_test,
    "test_size": 0.2,
    "random_state": 42,
    "split_method": "stratified_holdout_80_20"
}

joblib.dump(split_data, SPLIT_PKL)
print(f"[HOLDOUT] Arquivo salvo em: {SPLIT_PKL.resolve()}")
