# knn_runner.py
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# ================== CONFIG: pastas dos conjuntos ==================
DIR_ORIG_2F = "AtividadePraticaKNN/Dados_Originais_2Features"
DIR_NORM_2F = "AtividadePraticaKNN/Dados_Normalizados_2Features"
DIR_ORIG_11F = "AtividadePraticaKNN/Dados_Originais_11Features"
DIR_NORM_11F = "AtividadePraticaKNN/Dados_Normalizados_11Features"
CLASS_COL_CANDIDATES = ["class", "target", "label"]

# ================== Helpers de IO ==================
def smart_read(path: str) -> pd.DataFrame:
    """Tenta tab, vírgula e ponto-e-vírgula; cai para detecção automática."""
    for sep in ["\t", ",", ";"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
    return pd.read_csv(path, engine="python")

def find_train_test(dir_rel: str):
    """Acha train/test por nome; se não achar, usa os dois primeiros arquivos."""
    base = Path(__file__).resolve().parent
    dir_abs = (base / dir_rel).resolve()
    if not dir_abs.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {dir_abs}")

    files = sorted(glob.glob(str(dir_abs / "*.txt"))) + sorted(glob.glob(str(dir_abs / "*.csv")))
    if len(files) < 2:
        raise FileNotFoundError(f"Esperava ao menos 2 arquivos (train/test) em {dir_abs}. Achados: {files}")

    lower = [os.path.basename(p).lower() for p in files]
    train_idx = next((i for i, n in enumerate(lower) if "train" in n or "training" in n), None)
    test_idx  = next((i for i, n in enumerate(lower) if "test" in n or "testing" in n), None)

    if train_idx is not None and test_idx is not None:
        train_path, test_path = files[train_idx], files[test_idx]
    else:
        train_path, test_path = files[0], files[1]

    return train_path, test_path

def split_Xy(df: pd.DataFrame):
    """Detecta ID e class (case-insensitive)."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    lower_map = {c.lower(): c for c in df.columns}

    id_col = lower_map.get("id")
    class_col = None
    for cand in CLASS_COL_CANDIDATES:
        if cand in lower_map:
            class_col = lower_map[cand]
            break

    if id_col is None:
        raise ValueError(f"Coluna 'ID' não encontrada. Colunas: {list(df.columns)}")
    if class_col is None:
        raise ValueError(f"Coluna 'class/target/label' não encontrada. Colunas: {list(df.columns)}")

    X = df.drop(columns=[id_col, class_col])
    y = df[class_col].astype(int)
    ids = df[id_col].astype(str)
    return X, y, ids, id_col, class_col

def print_block(label: str, k: int, acc: float, y_true, y_pred, test_ids, neigh_ids_all, dists_all):
    print("\n" + "=" * 80)
    print(f"[A] {label} | k={k}")
    print("=" * 80)
    print(f"Acurácia: {acc:.3f}")
    print("Predições por instância de teste:")
    for i, tid in enumerate(test_ids):
        print(f"  Teste {tid}: y_true={int(y_true[i])} | y_pred={int(y_pred[i])}")
        print("    k-vizinhos (ID, dist):")
        for nid, dd in zip(neigh_ids_all[i], dists_all[i]):
            print(f"      {nid}  (d={dd:.4f})")

# ================== Perguntas ao usuário ==================
print("\nSelecione o modelo (um por vez):")
print("  1) 2 features - Originais")
print("  2) 2 features - Normalizados")
print("  3) 11 features - Originais")
print("  4) 11 features - Normalizados")
choice = input("Digite 1, 2, 3 ou 4: ").strip()

if choice == "1":
    dir_choice = DIR_ORIG_2F
    label = "Originais 2F"
elif choice == "2":
    dir_choice = DIR_NORM_2F
    label = "Normalizados 2F"
elif choice == "3":
    dir_choice = DIR_ORIG_11F
    label = "Originais 11F"
elif choice == "4":
    dir_choice = DIR_NORM_11F
    label = "Normalizados 11F"
else:
    raise SystemExit("Opção inválida.")

k = int(input("Informe o valor de k (ex.: 1,3,5,7): ").strip())
if k <= 0:
    raise SystemExit("k deve ser inteiro positivo.")

# ================== Carregar dados ==================
train_path, test_path = find_train_test(dir_choice)
print(f"\n[OK] Carregado: {os.path.basename(train_path)} (train) | {os.path.basename(test_path)} (test)")
train_df = smart_read(train_path)
test_df  = smart_read(test_path)

Xtr, ytr, id_tr, id_col, class_col = split_Xy(train_df)
Xte, yte, id_te, _, _ = split_Xy(test_df)

# ================== Treinar, prever e coletar vizinhos ==================
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(Xtr, ytr)
y_pred = knn.predict(Xte)
acc = accuracy_score(yte, y_pred)

# vizinhos para TODAS as instâncias de teste
dists, idxs = knn.kneighbors(Xte, n_neighbors=k, return_distance=True)
id_tr_reset = id_tr.reset_index(drop=True)
neigh_ids_all = [[id_tr_reset.iloc[j] for j in row] for row in idxs]

# ================== Impressão no formato solicitado ==================
print_block(label, k, acc, y_true=yte.values, y_pred=y_pred, test_ids=id_te.values,
            neigh_ids_all=neigh_ids_all, dists_all=dists)

# ================== Matriz de confusão (extra, útil para checar) ==================
cm = confusion_matrix(yte, y_pred)
print("\nMatriz de confusão (linhas=verdadeiro, colunas=previsto):")
print(cm)

# ================== Plot (2D direto para 2F; PCA-2D para 11F) ==================
fig, ax = plt.subplots(figsize=(7.5, 6.0))
if Xtr.shape[1] == 2:
    feat_names = list(Xtr.columns)
    ax.scatter(Xtr.iloc[:, 0], Xtr.iloc[:, 1], c=ytr, alpha=0.7, label="Treino")
    ax.scatter(Xte.iloc[:, 0], Xte.iloc[:, 1], marker="*", s=180,
               edgecolors="black", linewidths=1.2, c=y_pred, label="Teste (pred.)")
    for i, _id in enumerate(id_te):
        ax.annotate(str(_id), (Xte.iloc[i, 0], Xte.iloc[i, 1]), fontsize=8, xytext=(3,3), textcoords='offset points')
    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1])
else:
    pca = PCA(n_components=2, random_state=42)
    pca.fit(Xtr)
    Ztr = pca.transform(Xtr)
    Zte = pca.transform(Xte)
    ax.scatter(Ztr[:, 0], Ztr[:, 1], c=ytr, alpha=0.7, label="Treino")
    ax.scatter(Zte[:, 0], Zte[:, 1], marker="*", s=180,
               edgecolors="black", linewidths=1.2, c=y_pred, label="Teste (pred.)")
    for i, _id in enumerate(id_te):
        ax.annotate(str(_id), (Zte[i, 0], Zte[i, 1]), fontsize=8, xytext=(3,3), textcoords='offset points')
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

ax.set_title(f"kNN ({label}) — k={k} • acc={acc:.3f}")
ax.legend(loc="best")
plt.tight_layout()
out_png = f"knn_plot_{label.replace(' ', '_')}_k{k}.png"
plt.savefig(out_png, dpi=150)
print(f"\nFigura salva em: {out_png}")
plt.show()
