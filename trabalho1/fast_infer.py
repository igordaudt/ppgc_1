# fast_infer.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import typing as t

import numpy as np
import pandas as pd
import joblib

# === Caminhos de artefatos ===
VECTORIZER_PKL = Path("./trabalho1/cenario/tfidf_vectorizer.pkl")  # vetorizar
TUNED_PKL      = Path("./trabalho1/cenario/nb_tuned_model.pkl")    # modelo tunado
BASE_PKL       = Path("./trabalho1/cenario/nb_model.pkl")          # modelo baseline

# Dataset limpo para mapear Sub_Name <-> ID_Sub (opcional, mas recomendado)
CLEAN_TSV      = Path("./trabalho1/cenario/materiais_clean_fixed.tsv")


# ---------- Carregamento otimizado (cache) ----------
@lru_cache(maxsize=1)
def _load_vectorizer_and_model():
    """Carrega (uma única vez) o vectorizer e o melhor modelo disponível (tuned > baseline)."""
    if not VECTORIZER_PKL.exists():
        raise FileNotFoundError(f"Vectorizer não encontrado: {VECTORIZER_PKL.resolve()}")

    vectorizer = joblib.load(VECTORIZER_PKL)

    model_pack = None
    model_source = None
    if TUNED_PKL.exists():
        model_pack = joblib.load(TUNED_PKL)
        model_source = "tuned"
    elif BASE_PKL.exists():
        model_pack = joblib.load(BASE_PKL)
        model_source = "baseline"
    else:
        raise FileNotFoundError(
            f"Modelo não encontrado. Esperado um dos arquivos: {TUNED_PKL} ou {BASE_PKL}"
        )

    model = model_pack["model"]
    classes = model_pack.get("classes_", getattr(model, "classes_", None))
    if classes is None:
        classes = getattr(model, "classes_", None)
    classes = np.array(classes)

    # detecta se há predict_proba (usaremos para decidir rapidamente caminho de inferência)
    has_proba = hasattr(model, "predict_proba")

    return vectorizer, model, classes, model_source, has_proba


@lru_cache(maxsize=1)
def _label_maps():
    """
    Retorna dois dicionários para mapear rótulos entre ID e Nome:
      - name_to_id:  Sub_Name -> ID_Sub (mais frequente)
      - id_to_name:  ID_Sub   -> Sub_Name (mais frequente)
    Se não houver dataset limpo, retorna dicionários vazios.
    """
    if not CLEAN_TSV.exists():
        return {}, {}

    df = pd.read_csv(CLEAN_TSV, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)

    name_to_id = {}
    id_to_name = {}

    if "Sub_Name" in df.columns and "ID_Sub" in df.columns:
        if len(df):
            name_to_id = (
                df[df["Sub_Name"].astype(str).str.len() > 0]
                .groupby("Sub_Name")["ID_Sub"]
                .agg(lambda s: s.value_counts().idxmax())
                .to_dict()
            )
            id_to_name = (
                df[df["ID_Sub"].astype(str).str.len() > 0]
                .groupby("ID_Sub")["Sub_Name"]
                .agg(lambda s: s.value_counts().idxmax())
                .to_dict()
            )
    return name_to_id, id_to_name


def _resolve_id_and_name(pred_label: str) -> t.Tuple[t.Optional[str], t.Optional[str]]:
    """
    Dado o rótulo que o modelo devolveu (pode ser Sub_Name OU ID_Sub),
    retorna (ID_Sub, Sub_Name) resolvidos via mapas. Se não conseguir resolver,
    retorna (None, None) e o chamador pode usar 'pred_label' como fallback.
    """
    name_to_id, id_to_name = _label_maps()

    # Caso 1: modelo devolveu nome da subcategoria
    if pred_label in name_to_id:
        return name_to_id[pred_label], pred_label

    # Caso 2: modelo devolveu ID
    if pred_label in id_to_name:
        return pred_label, id_to_name[pred_label]

    # Caso 3: modelo devolveu algo numérico que exista como ID (com zeros à esquerda?)
    if pred_label.isdigit() and pred_label in id_to_name:
        return pred_label, id_to_name[pred_label]

    # Não foi possível resolver
    return None, None


# ---------- Função principal otimizada ----------
def classify_products(
    terms: t.Union[str, t.List[str]],
    return_prob: bool = False,
    top_k: int = 1
) -> dict:
    """
    Classifica rapidamente descrições de produtos em subcategorias.

    Parâmetros:
      - terms: string única OU lista de strings.
      - return_prob: se True, inclui probabilidade/score no retorno.
      - top_k: se >1, inclui alternativas top-k (labels & probas).

    Retorno:
      Dicionário {descricao: {"ID_Sub": str|None, "Sub_Name": str|None, ...}}
      (Se não conseguir resolver o par ID/Nome pelos mapas, retorna os melhores possíveis
       e sempre inclui 'predicted_label' para auditoria.)
    """
    vectorizer, model, classes, model_source, has_proba = _load_vectorizer_and_model()

    # normalizar entrada
    if isinstance(terms, str):
        texts = [terms.strip()]
    else:
        texts = [str(t).strip() for t in terms]

    # vetorização em lote (rápida)
    X = vectorizer.transform(texts)

    # predição
    if has_proba:
        proba = model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = classes[pred_idx]
        pred_scores = proba[np.arange(len(texts)), pred_idx]
    else:
        pred_labels = model.predict(X)
        pred_scores = np.full(len(texts), np.nan, dtype=float)
        # criar matriz "proba" dummy para top-k se solicitado
        if top_k and top_k > 1:
            proba = np.zeros((len(texts), len(classes)), dtype=float)
            proba[np.arange(len(texts)), [np.where(classes == lab)[0][0] for lab in pred_labels]] = 1.0
        else:
            proba = None

    # montar resultado
    results = {}
    for i, desc in enumerate(texts):
        label = str(pred_labels[i])
        id_sub, sub_name = _resolve_id_and_name(label)

        # fallbacks caso não exista mapa:
        if id_sub is None and sub_name is None:
            # se o label parece numérico, tratamos como ID
            if label.isdigit():
                id_sub = label
                sub_name = None
            else:
                sub_name = label
                id_sub = None

        item = {
            "ID_Sub": id_sub,
            "Sub_Name": sub_name,
            "predicted_label": label,     # sempre bom manter para auditoria
            "model_source": model_source  # 'tuned' ou 'baseline'
        }

        if return_prob:
            item["score"] = float(pred_scores[i]) if not np.isnan(pred_scores[i]) else None

        if top_k and top_k > 1 and proba is not None:
            k = min(top_k, proba.shape[1])
            order = np.argsort(-proba[i])[:k]
            item["top_k_labels"] = [str(classes[j]) for j in order]
            item["top_k_scores"] = [float(proba[i, j]) for j in order]

        results[desc] = item

    return results


# ---------- Prompt interativo ----------
if __name__ == "__main__":
    import time

    print("=== Classificador de Produtos ===")
    print("Digite descrições de produtos para classificar.")
    print("Pressione ENTER vazio para sair.\n")

    while True:
        termo = input("Descrição do produto: ").strip()
        if not termo:
            print("Encerrando.")
            break

        start = time.perf_counter()
        res = classify_products(termo, return_prob=True, top_k=3)
        elapsed = time.perf_counter() - start
        r = res[termo]

        print("\n=== Resultado ===")
        print(f"Descrição     : {termo}")
        print(f"ID_Sub        : {r.get('ID_Sub')}")
        print(f"Subcategoria  : {r.get('Sub_Name')}")
        print(f"Predição      : {r.get('predicted_label')}  "
              f"(modelo: {r.get('model_source')})")
        if 'score' in r and r['score'] is not None:
            print(f"Confiança     : {r['score']:.3f}")
        if 'top_k_labels' in r:
            print("Top-3 opções  :")
            for lbl, sc in zip(r['top_k_labels'], r['top_k_scores']):
                print(f"   - {lbl:25s}  p={sc:.3f}")
        print(f"Tempo resposta: {elapsed:.4f} segundos\n")
