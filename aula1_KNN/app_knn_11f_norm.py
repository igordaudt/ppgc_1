import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ============ CONFIG ============
DIR_NORM_11F = "AtividadePraticaKNN/Dados_Normalizados_11Features"
CITRIC_COL = "citric.acid"           # ajuste se o nome for diferente no seu arquivo
CLASS_COL_CANDIDATES = ["class", "target", "label"]

st.set_page_config(page_title="KNN - 11 Features (Normalizado)", layout="wide")
st.title("KNN - Dados Normalizados (11 features) • Visualização e Experimentos")

# ============ IO HELPERS ============
def smart_read(path: str) -> pd.DataFrame:
    """Tenta ler TXT/CSV com separadores comuns (\t, ',' e ';')."""
    for sep in ["\t", ",", ";"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
    # última tentativa: detecção automática do pandas
    return pd.read_csv(path, engine="python")

def find_train_test(dir_rel: str):
    """Encontra arquivos de train/test pela heurística de nome; fallback = 2 primeiros."""
    base = Path(__file__).resolve().parent
    dir_abs = (base / dir_rel).resolve()
    if not dir_abs.exists():
        st.error(f"Pasta não encontrada: {dir_abs}")
        st.stop()

    files = sorted(glob.glob(str(dir_abs / "*.txt"))) + sorted(glob.glob(str(dir_abs / "*.csv")))
    if len(files) < 2:
        st.error(f"Esperava ao menos 2 arquivos (train/test) em {dir_abs}. Achados: {files}")
        st.stop()

    lower = [os.path.basename(p).lower() for p in files]
    train_idx = next((i for i, n in enumerate(lower) if "train" in n or "training" in n), None)
    test_idx  = next((i for i, n in enumerate(lower) if "test" in n or "testing" in n), None)

    if train_idx is not None and test_idx is not None:
        train_path, test_path = files[train_idx], files[test_idx]
    else:
        train_path, test_path = files[0], files[1]

    return train_path, test_path

def split_Xy(df: pd.DataFrame):
    """Detecta ID/class de forma case-insensitive; o resto é X."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    lower_map = {c.lower(): c for c in df.columns}

    id_col = lower_map.get("id")
    class_col = None
    for cand in CLASS_COL_CANDIDATES:
        if cand in lower_map:
            class_col = lower_map[cand]
            break

    if id_col is None:
        st.error(f"Coluna 'ID' não encontrada. Colunas: {list(df.columns)}")
        st.stop()
    if class_col is None:
        st.error(f"Coluna 'class' (ou target/label) não encontrada. Colunas: {list(df.columns)}")
        st.stop()

    X = df.drop(columns=[id_col, class_col])
    y = df[class_col].astype(int)
    ids = df[id_col].astype(str)
    return X, y, ids, id_col, class_col

# ============ LOAD ============
train_path, test_path = find_train_test(DIR_NORM_11F)

c1, c2 = st.columns([1, 1])
with c1:
    st.caption("Arquivo de **treino**")
    st.code(os.path.basename(train_path))
with c2:
    st.caption("Arquivo de **teste**")
    st.code(os.path.basename(test_path))

train_df = smart_read(train_path)
test_df  = smart_read(test_path)

st.subheader("Prévia dos dados")
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("**Treino**")
    st.dataframe(train_df.head())
with cc2:
    st.markdown("**Teste**")
    st.dataframe(test_df.head())

# ============ CONTROLES ============
st.sidebar.header("Configurações do KNN")
k = st.sidebar.select_slider("k (n_neighbors)", options=[1, 3, 5, 7], value=5)

Xtr, ytr, id_tr, id_col, class_col = split_Xy(train_df)
Xte, yte, id_te, _, _ = split_Xy(test_df)

feature_names = list(Xtr.columns)
st.sidebar.markdown("### Eixos do gráfico (2D)")
x_feat = st.sidebar.selectbox("Eixo X", feature_names, index=0)
y_feat = st.sidebar.selectbox("Eixo Y", feature_names, index=1 if len(feature_names) > 1 else 0)

# ============ MODELO ============
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(Xtr, ytr)
ypred = knn.predict(Xte)
acc = accuracy_score(yte, ypred)
st.metric("Acurácia no conjunto de teste", f"{acc:.3f}")

# ============ INSPEÇÃO ============
st.subheader("Inspecionar instância de teste e vizinhos (k-NN)")
ids_disponiveis = list(id_te)
default_id = ids_disponiveis[0] if ids_disponiveis else None
chosen_id = st.selectbox("Instância de teste", ids_disponiveis, index=ids_disponiveis.index(default_id))

row = test_df[test_df[id_col].astype(str) == str(chosen_id)].iloc[0]
xi = row.drop(labels=[id_col, class_col]).to_frame().T

dists, idxs = knn.kneighbors(xi, n_neighbors=k, return_distance=True)
id_tr_reset = id_tr.reset_index(drop=True)
viz_ids = [id_tr_reset.iloc[i] for i in idxs[0]]

y_true = int(row[class_col])
y_pred = int(knn.predict(xi)[0])

c3, c4 = st.columns([1, 1])
with c3:
    st.write(f"**Predição original** para `{chosen_id}`: `y_true={y_true}` • `y_pred={y_pred}`")
with c4:
    viz_table = pd.DataFrame({"neighbor_id": viz_ids, "distance": dists[0]})
    st.markdown("**Vizinhos (ID, distância)**")
    st.dataframe(viz_table)

# ============ GRÁFICO 2D ============
st.subheader("Dispersão 2D (selecione as features na barra lateral)")
fig, ax = plt.subplots()
ax.scatter(Xtr[x_feat], Xtr[y_feat], c=ytr, alpha=0.7, label="Treino")
ax.scatter(xi[x_feat].iloc[0], xi[y_feat].iloc[0],
           marker="*", s=200, edgecolors="black", linewidths=1.5, label=f"Teste {chosen_id}")
ax.set_xlabel(x_feat)
ax.set_ylabel(y_feat)
ax.legend(loc="best")
st.pyplot(fig)

# ============ PERTURBAÇÃO citric.acid ============
st.subheader("Perturbar `citric.acid` (no espaço normalizado)")
if CITRIC_COL not in xi.columns:
    st.info(f"A coluna `{CITRIC_COL}` não existe nesse conjunto. Colunas: {list(xi.columns)}")
else:
    colA, colB = st.columns(2)
    with colA:
        new_val = st.slider("Novo valor para citric.acid", 0.0, 1.0, float(xi[CITRIC_COL].iloc[0]), step=0.01)
    with colB:
        st.caption("Atalhos")
        if st.button("citric.acid = 0.30"):
            new_val = 0.30
        if st.button("citric.acid = 0.85"):
            new_val = 0.85

    x_syn = xi.copy()
    x_syn.loc[:, CITRIC_COL] = new_val

    d_p, i_p = knn.kneighbors(x_syn, n_neighbors=k, return_distance=True)
    ids_p = [id_tr_reset.iloc[i] for i in i_p[0]]
    y_pred_p = int(knn.predict(x_syn)[0])

    st.write(f"**Nova predição** com `citric.acid={new_val:.2f}`: `y_pred={y_pred_p}`")
    viz_table_p = pd.DataFrame({"neighbor_id": ids_p, "distance": d_p[0]})
    st.dataframe(viz_table_p)

    # destaca ponto perturbado nos eixos escolhidos
    fig2, ax2 = plt.subplots()
    ax2.scatter(Xtr[x_feat], Xtr[y_feat], c=ytr, alpha=0.7, label="Treino")
    ax2.scatter(xi[x_feat].iloc[0], xi[y_feat].iloc[0],
                marker="*", s=160, edgecolors="black", linewidths=1.2, label=f"Teste {chosen_id} (orig.)")
    # se x/y são o próprio citric.acid, o deslocamento aparecerá no eixo
    # caso contrário, o ponto perturbado não muda de posição visual (mas a classificação muda)
    ax2.scatter(x_syn[x_feat].iloc[0], x_syn[y_feat].iloc[0],
                marker="X", s=160, edgecolors="black", linewidths=1.2,
                label=f"Perturbado ({CITRIC_COL}={new_val:.2f})")
    ax2.set_xlabel(x_feat)
    ax2.set_ylabel(y_feat)
    ax2.legend(loc="best")
    st.pyplot(fig2)

st.caption("Dica: troque as features do gráfico na barra lateral, ajuste k e teste perturbações de `citric.acid`.")
