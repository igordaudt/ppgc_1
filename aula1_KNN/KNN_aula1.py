# knn_runner_simple.py
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Pastas dos dados
DIR_ORIG_2F = "AtividadePraticaKNN/Dados_Originais_2Features"
DIR_NORM_2F = "AtividadePraticaKNN/Dados_Normalizados_2Features"
DIR_ORIG_11F = "AtividadePraticaKNN/Dados_Originais_11Features"
DIR_NORM_11F = "AtividadePraticaKNN/Dados_Normalizados_11Features"

# Lê arquivo tentando separadores comuns
def read_any(path):
    for sep in ["\t", ",", ";"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            pass
    return pd.read_csv(path, engine="python")

# Acha train/test (prioriza nomes com "train"/"test")
def find_train_test(dir_rel):
    base = Path(__file__).resolve().parent
    d = (base / dir_rel).resolve()
    files = sorted(glob.glob(str(d / "*.txt"))) + sorted(glob.glob(str(d / "*.csv")))
    if len(files) < 2:
        raise FileNotFoundError(f"Arquivos insuficientes em {d}")
    low = [os.path.basename(p).lower() for p in files]
    i_tr = next((i for i,n in enumerate(low) if "train" in n or "training" in n), None)
    i_te = next((i for i,n in enumerate(low) if "test" in n or "testing" in n), None)
    if i_tr is not None and i_te is not None:
        return files[i_tr], files[i_te]
    return files[0], files[1]

# Separa X, y e IDs (case-insensitive para "ID" e "class")
def split_Xy(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    low = {c.lower(): c for c in df.columns}
    id_col = low.get("id")
    cls_col = low.get("class") or low.get("target") or low.get("label")
    if not id_col or not cls_col:
        raise ValueError(f"Esperado colunas ID e class. Colunas: {list(df.columns)}")
    X = df.drop(columns=[id_col, cls_col])
    y = df[cls_col].astype(int)
    ids = df[id_col].astype(str)
    return X, y, ids

# Impressão no formato solicitado
def print_block(label, k, acc, y_true, y_pred, test_ids, neigh_ids_all, dists_all):
    print("\n" + "="*80)
    print(f"[A] {label} | k={k}")
    print("="*80)
    print(f"Acurácia: {acc:.3f}")
    print("Predições por instância de teste:")
    for i, tid in enumerate(test_ids):
        print(f"  Teste {tid}: y_true={int(y_true[i])} | y_pred={int(y_pred[i])}")
        print("    k-vizinhos (ID, dist):")
        for nid, dd in zip(neigh_ids_all[i], dists_all[i]):
            print(f"      {nid}  (d={dd:.4f})")

# Menu simples
print("\nSelecione o modelo:")
print("  1) 2F - Originais")
print("  2) 2F - Normalizados")
print("  3) 11F - Originais")
print("  4) 11F - Normalizados")
opt = input("Digite 1, 2, 3 ou 4: ").strip()
if opt == "1":
    data_dir, label = DIR_ORIG_2F, "Originais 2F"
elif opt == "2":
    data_dir, label = DIR_NORM_2F, "Normalizados 2F"
elif opt == "3":
    data_dir, label = DIR_ORIG_11F, "Originais 11F"
elif opt == "4":
    data_dir, label = DIR_NORM_11F, "Normalizados 11F"
else:
    raise SystemExit("Opção inválida.")

k = int(input("Informe k (ex.: 1,3,5,7): ").strip())

# Carrega dados
train_path, test_path = find_train_test(data_dir)
print(f"\n[OK] Carregado: {os.path.basename(train_path)} (train) | {os.path.basename(test_path)} (test)")
train_df = read_any(train_path)
test_df  = read_any(test_path)

Xtr, ytr, id_tr = split_Xy(train_df)
Xte, yte, id_te = split_Xy(test_df)

# Treino e predição
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(Xtr, ytr)
y_pred = knn.predict(Xte)
acc = accuracy_score(yte, y_pred)

# Vizinhos do teste
dists, idxs = knn.kneighbors(Xte, n_neighbors=k, return_distance=True)
id_tr_reset = id_tr.reset_index(drop=True)
neigh_ids_all = [[id_tr_reset.iloc[j] for j in row] for row in idxs]

# Saída no padrão
print_block(label, k, acc, y_true=yte.values, y_pred=y_pred, test_ids=id_te.values,
            neigh_ids_all=neigh_ids_all, dists_all=dists)

# Plot: 2F direto; 11F usa PCA para 2D
plt.figure(figsize=(7.2, 5.8))
if Xtr.shape[1] == 2:
    names = list(Xtr.columns)
    plt.scatter(Xtr.iloc[:,0], Xtr.iloc[:,1], c=ytr, alpha=0.7, label="Treino")
    plt.scatter(Xte.iloc[:,0], Xte.iloc[:,1], marker="*", s=180, edgecolors="black", linewidths=1.0,
                c=y_pred, label="Teste (pred.)")
    for i, _id in enumerate(id_te):
        plt.annotate(str(_id), (Xte.iloc[i,0], Xte.iloc[i,1]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.xlabel(names[0]); plt.ylabel(names[1])
else:
    pca = PCA(n_components=2, random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)
    plt.scatter(Ztr[:,0], Ztr[:,1], c=ytr, alpha=0.7, label="Treino")
    plt.scatter(Zte[:,0], Zte[:,1], marker="*", s=180, edgecolors="black", linewidths=1.0,
                c=y_pred, label="Teste (pred.)")
    for i, _id in enumerate(id_te):
        plt.annotate(str(_id), (Zte[i,0], Zte[i,1]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")

plt.title(f"kNN ({label}) — k={k} • acc={acc:.3f}")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
