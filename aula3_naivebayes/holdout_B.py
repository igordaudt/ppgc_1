import pandas as pd             # biblioteca para análise de dados
import matplotlib.pyplot as plt # biblioteca para visualização de informações
import seaborn as sns           # biblioteca para visualização de informações
import numpy as np              # biblioteca para operações com arrays multidimensionais
from sklearn.datasets import load_breast_cancer ## conjunto de dados a ser analisado
sns.set()

## Carregando os dados - Câncer de Mama
## https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

data = load_breast_cancer() ## carrega os dados de breast cancer
X = data.data  # matriz contendo os atributos
y = data.target  # vetor contendo a classe (0 para maligno e 1 para benigno) de cada instância
feature_names = data.feature_names  # nome de cada atributo
target_names = data.target_names  # nome de cada classe

print(f"Dimensões de X: {X.shape}\n")
print(f"Dimensões de y: {y.shape}\n")
print(f"Nomes dos atributos: {feature_names}\n")
print(f"Nomes das classes: {target_names}")


# Analisando o impacto do tamanho do conjunto de treino na avaliação de desempenho dos modelos

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Função para rodar o KNN com diferentes tamanhos de treino
def run_knn_analysis(X, y, test_size=0.1, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8], iterations=10):
    results = {}

    # Fixando o conjunto de teste
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Criando o objeto de normalização
    scaler = StandardScaler()
    scaler.fit(X_train_full)

    # Normalizando o conjunto de teste
    X_test = scaler.transform(X_test)

    for train_size in train_sizes:
        accuracies = []
        for _ in range(iterations):
            # Amostrando um conjunto de treino de tamanho variável
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train_full, y_train_full, train_size=train_size, random_state=None
            )

            # Normalizando o conjunto de treino amostrado
            X_train_sample = scaler.transform(X_train_sample)

            # Treinar o kNN
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(X_train_sample, y_train_sample)

            # Classificação e avaliação no conjunto de teste fixo
            y_pred_knn = knn_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_knn)
            accuracies.append(accuracy)

        # Calculando estatísticas do desempenho
        variance = np.var(accuracies)
        amplitude = np.max(accuracies) - np.min(accuracies)
        average = np.mean(accuracies)

        results[train_size] = {
            'accuracies': accuracies,
            'variance': variance,
            'amplitude': amplitude,
            'average': average
        }

    return results

# Exemplo de uso
variances = []
amplitudes = []
averages = []
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8]

results = run_knn_analysis(X, y)

for train_size, metrics in results.items():
    variances.append(metrics['variance'])
    amplitudes.append(metrics['amplitude'])
    averages.append(metrics['average'])

# Gráfico de variância vs tamanho do conjunto de treino
plt.figure(figsize=(10, 6))
plt.scatter(train_sizes, variances, c='blue', label='Variância do Desempenho')
plt.plot(train_sizes, variances, color='blue', linestyle='--')
plt.xlabel('Proporção do Conjunto de Treino')
plt.ylabel('Variância da Acurácia')
plt.title('Variância da Acurácia em Função do Tamanho do Conjunto de Treino')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de amplitude vs tamanho do conjunto de treino
plt.figure(figsize=(10, 6))
plt.scatter(train_sizes, amplitudes, c='red', label='Amplitude do Desempenho (Max - Min)')
plt.plot(train_sizes, amplitudes, color='red', linestyle='--')
plt.xlabel('Proporção do Conjunto de Treino')
plt.ylabel('Amplitude da Acurácia')
plt.title('Amplitude da Acurácia em Função do Tamanho do Conjunto de Treino')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de média vs tamanho do conjunto de treino
plt.figure(figsize=(10, 6))
plt.scatter(train_sizes, averages, c='b', label='Média')
plt.plot(train_sizes, averages, color='b', linestyle='--')
plt.xlabel('Proporção do Conjunto de Treino')
plt.ylabel('Média da Acurácia')
plt.title('Média da Acurácia em Função do Tamanho do Conjunto de Treino')
plt.legend()
plt.grid(True)
plt.show()