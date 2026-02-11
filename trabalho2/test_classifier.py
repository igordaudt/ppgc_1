# svm_predict_cli.py
# Uso interativo do modelo SVM treinado para classificar materiais de construção
# (Trabalho 2 - Aprendizado de Máquina)

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ========================== CONFIGURAÇÕES ==========================

# ⚠️ AJUSTE OS CAMINHOS ABAIXO CONFORME O SEU PROJETO ⚠️
VECTORIZER_PKL = Path("./trabalho2/tf-idf/tfidf_vectorizer.pkl")
MODEL_PKL      = Path("./trabalho2/modelos/svm_model.pkl")

# Opcional: arquivo com o dicionário de subcategorias (ID_Sub -> nome)
# Se não existir, o código continua funcionando apenas com o ID da classe.
SUBMAP_TSV     = Path("./trabalho2/dados/subcategorias.tsv")

MODEL_NAME = "Support Vector Machine (SVM)"


# ========================== 1. CARREGAR ARTEFATOS ==========================

def load_vectorizer():
    if not VECTORIZER_PKL.exists():
        raise FileNotFoundError(f"Vectorizer não encontrado em: {VECTORIZER_PKL.resolve()}")
    vectorizer = joblib.load(VECTORIZER_PKL)
    return vectorizer


def load_model():
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Modelo SVM não encontrado em: {MODEL_PKL.resolve()}")
    model = joblib.load(MODEL_PKL)
    return model


def load_subcategory_map():
    """
    Tenta carregar um mapa ID_Sub -> Sub_Name.
    Espera um TSV/CSV com pelo menos as colunas:
       - ID_Sub
       - Sub_Name  (nome legível da subcategoria)
    Se o arquivo não existir, retorna None.
    """
    if not SUBMAP_TSV.exists():
        return None

    df = pd.read_csv(SUBMAP_TSV, sep="\t", encoding="utf-8")
    # Ajuste os nomes das colunas se forem diferentes
    if "ID_Sub" not in df.columns or "Sub_Name" not in df.columns:
        return None

    return dict(zip(df["ID_Sub"], df["Sub_Name"]))


# ========================== 2. CLASSIFICAÇÃO ==========================

def _softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Converte scores arbitrários (ex.: decision_function do SVM)
    em pseudo-probabilidades via softmax, de forma estável numericamente.
    """
    # subtrai o máximo para evitar overflow numérico
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def classify_products(
    textos,
    return_prob: bool = True,
    top_k: int = 3,
):
    """
    Classifica um ou vários textos de produto usando o modelo SVM e o TF-IDF.

    Parâmetros
    ----------
    textos : str | list[str]
        Uma descrição ou uma lista de descrições de produtos.
    return_prob : bool
        Se True, calcula pseudo-probabilidades a partir do decision_function.
    top_k : int
        Número de rótulos mais prováveis a serem retornados.

    Retorno
    -------
    dict[str, dict]
        Dicionário mapeando cada texto de entrada para um dicionário de resultados.
    """
    # Garantir formato de lista
    single_input = False
    if isinstance(textos, str):
        textos = [textos]
        single_input = True

    # Carregar artefatos
    vectorizer = load_vectorizer()
    model = load_model()
    sub_map = load_subcategory_map()

    # Vetorização
    X = vectorizer.transform(textos)

    # Scores do SVM
    raw_scores = model.decision_function(X)  # shape (n_samples, n_classes) ou (n_samples,) no binário
    classes = model.classes_

    # Ajuste para o caso binário (decision_function retorna (n_samples,))
    if raw_scores.ndim == 1:
        # convenção: scores para [classe_negativa, classe_positiva]
        raw_scores = np.column_stack([-raw_scores, raw_scores])
        # garantir que 'classes' está na mesma ordem [neg, pos] (normalmente já está)
        # mas aqui assumimos que model.classes_ já está correto.

    resultados = {}

    for idx, texto in enumerate(textos):
        scores_sample = raw_scores[idx]  # array (n_classes,)

        # Top-k classes ordenadas por score decrescente
        order = np.argsort(scores_sample)[::-1]
        top_indices = order[:top_k]

        top_labels = classes[top_indices]
        top_scores = scores_sample[top_indices]

        # Probabilidades (softmax) — pseudo, não calibradas
        if return_prob:
            probs_all = _softmax_from_scores(scores_sample)
            top_probs = probs_all[top_indices]
        else:
            probs_all = None
            top_probs = None

        # Classe predita
        best_idx = top_indices[0]
        predicted_label = classes[best_idx]
        predicted_score = (
            float(top_probs[0]) if (return_prob and top_probs is not None) else float(top_scores[0])
        )

        # Mapear para subcategoria legível, se disponível
        # Aqui assumimos que 'predicted_label' é o ID_Sub.
        sub_name = None
        if sub_map is not None:
            sub_name = sub_map.get(predicted_label, None)

        # Montar resultado no mesmo estilo do NB
        resultados[texto] = {
            "ID_Sub": predicted_label,
            "Sub_Name": sub_name,
            "predicted_label": predicted_label,
            "model_source": MODEL_PKL.name,
            "score": predicted_score,
        }

        if return_prob:
            resultados[texto]["top_k_labels"] = top_labels.tolist()
            resultados[texto]["top_k_scores"] = top_probs.tolist()
        else:
            resultados[texto]["top_k_labels"] = top_labels.tolist()
            resultados[texto]["top_k_scores"] = top_scores.tolist()

    if single_input:
        # Mesmo para entrada única, mantemos o dicionário {texto: {...}}
        return resultados

    return resultados


# ========================== 3. PROMPT INTERATIVO ==========================

if __name__ == "__main__":
    print("=== Classificador de Produtos (SVM) ===")
    print("Digite descrições de produtos para classificar.")
    print("Pressione ENTER vazio para sair.\n")

    while True:
        termo = input("Descrição do produto: ").strip()
        if not termo:
            print("Encerrando.")
            break

        start = time.perf_counter()
        try:
            res = classify_products(termo, return_prob=True, top_k=3)
        except FileNotFoundError as e:
            print(f"\n[ERRO] {e}")
            print("Verifique os caminhos de VECTORIZER_PKL, MODEL_PKL e SUBMAP_TSV no início do arquivo.\n")
            break

        elapsed = time.perf_counter() - start
        r = res[termo]

        print("\n=== Resultado ===")
        print(f"Descrição     : {termo}")
        print(f"ID_Sub        : {r.get('ID_Sub')}")
        print(f"Subcategoria  : {r.get('Sub_Name')}")
        print(f"Predição      : {r.get('predicted_label')}  "
              f"(modelo: {r.get('model_source')})")

        if "score" in r and r["score"] is not None:
            print(f"Confiança(*)  : {r['score']:.3f}")
            print("  (*) pseudo-probabilidade derivada do decision_function (softmax).")

        if "top_k_labels" in r:
            print("Top-3 opções  :")
            for lbl, sc in zip(r["top_k_labels"], r["top_k_scores"]):
                print(f"   - {str(lbl):25s}  p≈{sc:.3f}")

        print(f"Tempo resposta: {elapsed:.4f} segundos\n")
