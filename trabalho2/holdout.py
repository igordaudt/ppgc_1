# trabalho2_holdout_split.py
# Divisao dos dados (holdout 80/20) para treino e teste

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ===================== CONFIGURACOES =====================
DATASET_PKL = Path("./trabalho2/tf-idf/tfidf_dataset.pkl")
HOLDOUT_DIR = Path("./trabalho2/holdout/")
SPLIT_PKL   = HOLDOUT_DIR / "train_test_split.pkl"
REMOVED_TXT = HOLDOUT_DIR / "removed_classes.txt"
MIN_CLASS_COUNT = 10

# ===================== 1. Carregar dataset TF-IDF =====================
print("[HOLDOUT] Carregando tfidf_dataset.pkl ...")
X_tfidf, y = joblib.load(DATASET_PKL)

print(f"[HOLDOUT] Total de amostras: {X_tfidf.shape[0]:,} | "
      f"Dimensoes TF-IDF: {X_tfidf.shape[1]:,} | "
      f"Classes: {len(set(y)):,}")

# ===================== 1.1. Ignorar classes mal representadas =====================
y_np = np.array(y)
labels, counts = np.unique(y_np, return_counts=True)
valid_labels = labels[counts >= MIN_CLASS_COUNT]
mask = np.isin(y_np, valid_labels)

X_tfidf = X_tfidf[mask]
y_np = y_np[mask]
y = y_np.tolist()

removed_labels = labels[counts < MIN_CLASS_COUNT]
removed_counts = counts[counts < MIN_CLASS_COUNT]

HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
with open(REMOVED_TXT, "w", encoding="utf-8") as f:
    f.write("label\tcount\n")
    for lbl, cnt in sorted(zip(removed_labels, removed_counts), key=lambda x: x[0]):
        f.write(f"{lbl}\t{cnt}\n")

print(f"[HOLDOUT] Classes com < {MIN_CLASS_COUNT} instancias removidas: {len(removed_labels)}")
print(f"[HOLDOUT] Lista salva em: {REMOVED_TXT}")
print(f"[HOLDOUT] Total apos filtragem: {X_tfidf.shape[0]:,} amostras | Classes: {len(valid_labels):,}")

# ===================== 2. Dividir em treino/teste =====================
# Estrategia: 80% treino, 20% teste, estratificado por classe
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"[HOLDOUT] Divisao concluida:")
print(f"  Treino: {X_train.shape[0]:,} amostras")
print(f"  Teste : {X_test.shape[0]:,} amostras")

# ===================== 3. Salvar resultados =====================
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
