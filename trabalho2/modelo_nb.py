# modelo_nb.py
# Classificação de materiais de construção usando Naive Bayes
# Estrutura padronizada conforme os demais modelos (Trabalho 2 - Aprendizado de Máquina)

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from time import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# ========================== CONFIGURAÇÕES ==========================

HOLDOUT_PKL = Path("./trabalho2/holdout/train_test_split.pkl")
MODEL_DIR   = Path("./trabalho2/modelos/")
METRIC_DIR  = Path("./trabalho2/metricas/")
FIG_DIR     = Path("./trabalho2/figuras/")

MODEL_PKL   = MODEL_DIR / "nb_model.pkl"
METRICS_TXT = METRIC_DIR / "nb_metrics.txt"
REPORT_TSV  = METRIC_DIR / "nb_class_report.tsv"
CM_PNG      = FIG_DIR / "nb_confusion_matrix.png"

MODEL_NAME  = "Naive Bayes"


# ========================== 1. CARREGAR DADOS ==========================

def load_holdout():
    """Carrega o conjunto de treino e teste a partir do arquivo de holdout."""
    if not HOLDOUT_PKL.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {HOLDOUT_PKL.resolve()}")
    
    split = joblib.load(HOLDOUT_PKL)
    X_train = split["X_train"]
    y_train = split["y_train"]
    X_test  = split["X_test"]
    y_test  = split["y_test"]

    print(f"[{MODEL_NAME}] Dados carregados:")
    print(f"  Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,} | Classes: {len(set(y_train)):,}")
    return X_train, X_test, y_train, y_test


# ========================== 2. CRIAR MODELO ==========================

def build_model():
    """Cria o modelo Naive Bayes Multinomial com hiperparâmetros padrão."""
    model = MultinomialNB(
        alpha=1.0,       # suavização de Laplace
        fit_prior=True
    )
    return model


# ========================== 3. TREINAMENTO ==========================

def train_model(model, X_train, y_train):
    """Treina o modelo e mede o tempo de execução."""
    print(f"[{MODEL_NAME}] Iniciando treinamento...")
    start = time()
    model.fit(X_train, y_train)
    end = time()
    print(f"[{MODEL_NAME}] Treinamento concluído em {end - start:.2f} segundos.")
    return model


# ========================== 4. AVALIAÇÃO ==========================

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo com métricas padrão e gera matriz de confusão."""
    print(f"[{MODEL_NAME}] Avaliando desempenho no conjunto de teste...")
    y_pred = model.predict(X_test)

    # Métricas globais
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0)
    }

    # Relatório detalhado (por classe)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    print(f"[{MODEL_NAME}] Avaliação concluída.")
    print(f"  Acurácia:        {metrics['accuracy']:.4f}")
    print(f"  F1-macro:        {metrics['f1_macro']:.4f}")
    print(f"  Precisão-macro:  {metrics['precision_macro']:.4f}")
    print(f"  Recall-macro:    {metrics['recall_macro']:.4f}")

    return metrics, report_df, cm


# ========================== 5. SALVAR RESULTADOS ==========================

def save_results(model, metrics, report_df, cm):
    """Salva modelo, métricas e figuras."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Salvar modelo treinado
    joblib.dump(model, MODEL_PKL)

    # Salvar métricas agregadas
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(f"{MODEL_NAME} - Resultados de Avaliação\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # Salvar relatório por classe
    report_df.to_csv(REPORT_TSV, sep="\t", encoding="utf-8-sig")

    # Preparar matriz para visualização (esconder zeros)
    cm_display = np.where(cm == 0, np.nan, cm)

    # Salvar matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 8, "color": "black"}
    )
    plt.title(f"{MODEL_NAME} - Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150)
    plt.close()

    print(f"[{MODEL_NAME}] Resultados salvos em:")
    print(f"  Modelo   : {MODEL_PKL}")
    print(f"  Métricas : {METRICS_TXT}")
    print(f"  Relatório: {REPORT_TSV}")
    print(f"  Matriz   : {CM_PNG}")


# ========================== 6. PIPELINE PRINCIPAL ==========================

def main():
    X_train, X_test, y_train, y_test = load_holdout()
    model = build_model()
    model = train_model(model, X_train, y_train)
    metrics, report_df, cm = evaluate_model(model, X_test, y_test)
    save_results(model, metrics, report_df, cm)
    print(f"\n[{MODEL_NAME}] Finalizado com F1-macro = {metrics['f1_macro']:.4f}")


# ========================== EXECUÇÃO ==========================

if __name__ == "__main__":
    main()
