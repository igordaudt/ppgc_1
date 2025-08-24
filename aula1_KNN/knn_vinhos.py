import os
import glob
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print ("Ajustando caminhos para os dados...")

# ========= CONFIG =========
# Ajuste estes caminhos para onde você descompactou cada conjunto:
DIR_ORIG_2F = r"./AtividadePraticaKNN/Dados_Originais_2Features"
DIR_NORM_2F = r"./AtividadePraticaKNN/Dados_Normalizados_2Features"
DIR_ORIG_11F = r"./AtividadePraticaKNN/Dados_Originais_11Features"
DIR_NORM_11F = r"./AtividadePraticaKNN/Dados_Normalizados_11Features"
# =========================

# Nomes esperados de colunas
ID_COL = "ID"
CLASS_COL = "class"
# Nome do atributo a ser perturbado (normalizado) no experimento C
CITRIC_COL = "citric acid"

# ========= FUNÇÕES AUX =========
def load_pair(dir_path: str):
    """
    Carrega automaticamente 1 TXT/CSV de treino e 1 de teste de um diretório.
    - Procura arquivos com 'train'/'training' e 'test'/'testing' no nome.
    - Caso contrário, assume o primeiro é train e o segundo é test.
    - Usa separador inteligente: tenta tab, vírgula e ponto e vírgula.
    Retorna: (train_df, test_df)
    """
    # busca tanto .txt quanto .csv
    csvs = sorted(glob.glob(os.path.join(dir_path, "*.txt"))) + \
           sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {dir_path}")

    # Heurística para identificar train/test
    train_cands = [p for p in csvs if any(k in os.path.basename(p).lower() for k in ["train", "training"])]
    test_cands  = [p for p in csvs if any(k in os.path.basename(p).lower() for k in ["test", "testing"])]

    if train_cands and test_cands:
        train_path, test_path = train_cands[0], test_cands[0]
    else:
        if len(csvs) < 2:
            raise ValueError(f"Esperava pelo menos 2 arquivos (train/test) em {dir_path}. Arquivos: {csvs}")
        train_path, test_path = csvs[0], csvs[1]

    def smart_read(path):
        for sep in ["\t", ",", ";"]:
            try:
                return pd.read_csv(path, sep=sep, engine="python")
            except Exception:
                continue
        raise ValueError(f"Não foi possível ler o arquivo {path}")

    train_df = smart_read(train_path)
    test_df  = smart_read(test_path)

    print(f"[OK] Carregado: {os.path.basename(train_path)} (train) | {os.path.basename(test_path)} (test)")
    return train_df, test_df


def split_Xy(df: pd.DataFrame):
    # normaliza espaços e maiúsc./minúsc.
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    lower_map = {c.lower(): c for c in df.columns}

    # detecta colunas de ID e classe por nome (case-insensitive)
    id_col = lower_map.get("id")
    class_col = lower_map.get("class") or lower_map.get("target") or lower_map.get("label")

    assert id_col is not None, "Coluna 'ID' não encontrada no arquivo"
    assert class_col is not None, "Coluna 'class' não encontrada no arquivo"

    X = df.drop(columns=[id_col, class_col])
    y = df[class_col].astype(int)
    ids = df[id_col].astype(str)

    return X, y, ids

def train_eval_knn(train_df, test_df, k):
    Xtr, ytr, id_tr = split_Xy(train_df)
    Xte, yte, id_te = split_Xy(test_df)
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(Xtr, ytr)
    ypred = knn.predict(Xte)
    acc = accuracy_score(yte, ypred)

    # também expõe vizinhos e distâncias para cada instância de teste
    dists, idxs = knn.kneighbors(Xte, n_neighbors=k, return_distance=True)
    # mapeia índices do treino para IDs do treino
    id_tr = id_tr.reset_index(drop=True)
    neighbors_ids = [[id_tr.iloc[i] for i in row] for row in idxs]

    return {
        "model": knn,
        "acc": acc,
        "ypred": ypred,
        "ytrue": yte.values,
        "test_ids": id_te.values,
        "neighbors_ids": neighbors_ids,
        "neighbors_dists": dists
    }

def print_run_summary(title, run_res):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"Acurácia: {run_res['acc']:.3f}")
    print("Predições por instância de teste:")
    for i, tid in enumerate(run_res["test_ids"]):
        print(f"  Teste {tid}: y_true={run_res['ytrue'][i]} | y_pred={run_res['ypred'][i]}")
        print("    k-vizinhos (ID, dist):")
        for nid, dd in zip(run_res["neighbors_ids"][i], run_res["neighbors_dists"][i]):
            print(f"      {nid}  (d={dd:.4f})")


def get_test_row(df_test, wanted_id: str):
    row = df_test[df_test[ID_COL].astype(str).str.upper() == wanted_id.upper()]
    if row.empty:
        raise ValueError(f"Instância de teste {wanted_id} não encontrada no teste.")
    return row.iloc[0]

# ========= EXPERIMENTO A =========
# k ∈ {1,3,5,7} em 2 atributos, com e sem normalização; comparar acurácias
orig2_train, orig2_test = load_pair(DIR_ORIG_2F)
norm2_train, norm2_test = load_pair(DIR_NORM_2F)

print("\n*** EXPERIMENTO A: 2 features (original vs normalizado), k in {1,3,5,7} ***")
for k in [1,3,5,7]:
    run_o = train_eval_knn(orig2_train, orig2_test, k)
    print_run_summary(f"[A] Originais 2F | k={k}", run_o)

    run_n = train_eval_knn(norm2_train, norm2_test, k)
    print_run_summary(f"[A] Normalizados 2F | k={k}", run_n)

# ========= EXPERIMENTO B =========
# Modelo k=5, 2 atributos, NÃO normalizado; listar k-vizinhos da N1
print("\n*** EXPERIMENTO B: k=5, 2F não normalizado; vizinhos da N1, N2, N3, N4 ***")
run_b = train_eval_knn(orig2_train, orig2_test, k=5)

# Para cada N1..N4, mostramos vizinhos do ponto alvo:
for wanted in ["N1", "N2", "N3", "N4"]:
    try:
        # Índice da linha desejada no conjunto de teste
        row = get_test_row(orig2_test, wanted)
        Xtr, ytr, id_tr = split_Xy(orig2_train)
        Xte, yte, id_te = split_Xy(orig2_test)
        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(Xtr, ytr)

        # posição da instância no Xte
        idx_test = orig2_test.index.get_loc(row.name)
        dist_w, idx_w = knn.kneighbors(Xte.iloc[[idx_test]], n_neighbors=5, return_distance=True)
        id_tr = id_tr.reset_index(drop=True)
        viz_ids = [id_tr.iloc[i] for i in idx_w[0]]

        print(f"\n[B] Vizinhos de {wanted}:")
        for nid, dd in zip(viz_ids, dist_w[0]):
            print(f"   {nid} (d={dd:.4f})")
    except Exception as e:
        print(f"\n[B] Aviso para {wanted}: {e}")

        # ========= EXPERIMENTO C =========
# k=5 em Normalizados_2Features e Normalizados_11Features;
# N4: prever, listar vizinhos; criar 2 instâncias sintéticas com citric acid = 0.3 e 0.85 e reclassificar.

print("\n*** EXPERIMENTO C: k=5 em dados NORMALIZADOS (2F e 11F), perturbar citric acid da N4 ***")
norm11_train, norm11_test = load_pair(DIR_NORM_11F)

def classify_and_neighbors_for_id(train_df, test_df, k, test_id, perturbs=None):
    """
    perturbs: lista de novos valores para CITRIC_COL (no espaço *normalizado*).
              Se None, só classifica a instância original.
    """
    Xtr, ytr, id_tr = split_Xy(train_df)
    Xte, yte, id_te = split_Xy(test_df)
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(Xtr, ytr)

    # localizar N4 original
    row = get_test_row(test_df, test_id)
    xi = row.drop(labels=[ID_COL, CLASS_COL]).to_frame().T
    yi_true = int(row[CLASS_COL])
    dists_o, idxs_o = knn.kneighbors(xi, n_neighbors=k, return_distance=True)
    pred_o = knn.predict(xi)[0]
    id_tr = id_tr.reset_index(drop=True)
    ids_o = [id_tr.iloc[i] for i in idxs_o[0]]

    print("\n[C] Modelo:", f"{train_df.shape[1]-2} atributos normalizados | k={k}")
    print(f"    Instância {test_id} original: y_true={yi_true} | y_pred={pred_o}")
    print("    k-vizinhos originais (ID, dist):")
    for nid, dd in zip(ids_o, dists_o[0]):
        print(f"      {nid}  (d={dd:.4f})")

    # Perturbações
    if perturbs:
        for val in perturbs:
            if CITRIC_COL not in xi.columns:
                print(f"    [!] Coluna '{CITRIC_COL}' não existe neste dataset; pulando perturbações.")
                break
            x_syn = xi.copy()
            x_syn.loc[:, CITRIC_COL] = val
            dists_p, idxs_p = knn.kneighbors(x_syn, n_neighbors=k, return_distance=True)
            pred_p = knn.predict(x_syn)[0]
            ids_p = [id_tr.iloc[i] for i in idxs_p[0]]
            print(f"    Perturbação: {CITRIC_COL}={val}")
            print(f"      y_pred={pred_p}")
            print("      k-vizinhos (ID, dist):")
            for nid, dd in zip(ids_p, dists_p[0]):
                print(f"        {nid}  (d={dd:.4f})")

# 2 FEATURES NORMALIZADOS
classify_and_neighbors_for_id(norm2_train, norm2_test, k=5, test_id="N4", perturbs=[0.3, 0.85])
# 11 FEATURES NORMALIZADOS
classify_and_neighbors_for_id(norm11_train, norm11_test, k=5, test_id="N4", perturbs=[0.3, 0.85])

print("\n=== FIM ===")