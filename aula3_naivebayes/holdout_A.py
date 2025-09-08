# Carregando as bibliotecas e dados
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

## transforma NumPy Array para Pandas DataFrame
data_df = pd.DataFrame(X,columns=feature_names)

## sumariza os atributos numéricos (todos, neste caso)
data_df.describe()

# Fazendo a divisão dos dados com Holdout de 3 vias (treino/validação/teste)

#Carregando funções específicas do scikit-learn

from sklearn.model_selection import train_test_split # função do scikit-learn que implementa um holdout
from sklearn.naive_bayes import GaussianNB # Naive Bayes Gaussiano
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report # métricas de desempenho
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # matriz de confusão

## Exemplo de HOLDOUT de 2 vias: separa os dados em treino e teste, de forma estratificada (não utilizado aqui)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y) ## atenção: inicialmente, não mude o random_state para este exercício

## HOLDOUT de 3 vias: separa os dados em treino e teste, de forma estratificada

## Definindo as proporções de treino, validação e teste.
train_ratio = 0.70
test_ratio = 0.15
validation_ratio = 0.15

## Fazendo a primeira divisão, para separar um conjunto de teste dos demais.
## Assuma X_temp e y_temp para os dados de treinamento+validação e X_test e y_test para os de teste
## Dica: configure o random_state para facilitar reprodutibilidade dos experimentos

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio,random_state=42,stratify=y)

## Fazendo a segunda divisão, para gerar o conjunto de treino e validação a partir
## do conjunto de 'treinamento' da divisão anterior
## Assuma X_train e y_train para os dados de treinamento e X_valid e y_valid para os de teste
## Dica: configure o random_state para facilitar reprodutibilidade dos experimentos

X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=validation_ratio/(train_ratio+test_ratio),random_state=42,stratify=y_temp)

print(X_train.shape) #retorna o shape da matriz de treino em linhas e colunas
print(X_test.shape) #retorna o shape da matriz de teste em linhas e colunas
print(X_valid.shape) #retorna o shape da matriz de validação em linhas e colunas


# Pré-processamento: Normalizando os dados
# A normalização é feita de forma a evitar Data Leakage (vazamento de informações dos dados de teste durante o treinamento dos modelos). Os parâmetros para normalização são estimados a partir dos dados de treino, e posteriormente aplicados para normalizar todos os dados, isto é, treino, validação e teste.

# A normalização é imprescindível para algoritmos baseados em distâncias, como o kNN.

from sklearn.preprocessing import MinMaxScaler # função do scikit-learn que implementa normalização min-max

## O MinMaxScaler transformará os dados para que fiquem no intervalo [0,1] - importante para o kNN
scaler = MinMaxScaler()

## Iniciar a normalização dos dados. Primeiro fazer um 'fit' do scaler nos
## dados de treino. Esta etapa visa "aprender" os parâmetros para normalização.
## No caso do MinMaxScales, são os valores mínimos e máximos de cada atributo
scaler.fit(X_train)

## Aplicar a normalização nos três conjuntos de dados:
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

##################################################################
# Treinando um modelo Naïve Bayes Gaussiano (para dados numéricos)

# Treinar Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Classificar dados no conjunto de teste
y_pred_nb = nb_model.predict(X_test)

# Avaliar o desempenho
print("Naive Bayes - Desempenho no Conjunto de Teste")
print(f"Acurácia: {accuracy_score(y_test, y_pred_nb):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_nb):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_nb):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_nb))

# Matriz de confusão
# Descomente as linhas abaixo para visualizar a matriz de confusão do Naive Bayes
# cm = confusion_matrix(y_test, y_pred_nb,labels=nb_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
# disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
# plt.grid(False)
# plt.show()

##################################################################
# Treinando um modelo kNN - com otimização do hiperparâmetro k
# Testar KNN com diferentes valores de k
# Conjunto de validação é usado para selecionar o melhor k
# Conjunto de teste é usado para avaliação final do modelo otimizado

# A análise é feita com a distância Euclidiana (padrão)
best_k = 1
best_score = 0

for k in range(1, 17,2):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Classificar dados no conjunto de teste
    y_pred_valid = knn_model.predict(X_valid)

    # Acurácia no conjunto de validação
    score = accuracy_score(y_valid, y_pred_valid)
    print(f"K={k}: Acurácia na Validação = {score:.2f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\nMelhor valor de K: {best_k} com Acurácia de {best_score:.2f} na Validação")

# Avaliação final do KNN com o melhor k
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train) ## o ideal seria unir treino+validação neste treinamento, mas para fins de comparação entre modelos knn/NB mantive apenas X_train

# Classificar dados no conjunto de teste
y_pred_knn = knn_model.predict(X_test)

print(f"\nKNN - Desempenho com K={best_k} no Conjunto de Teste")
print(f"Acurácia: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_knn):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_knn):.2f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_knn))

# Matriz de confusão
# Descomente as linhas abaixo para visualizar a matriz de confusão do kNN
# cm = confusion_matrix(y_test, y_pred_knn,labels=knn_model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
# disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
# plt.grid(False)
# plt.show()


##################################################################
# Analisando o impacto da divisão aleatória de dados no desempenho dos modelos
# Inicializar listas para armazenar os resultados
accuracies = []
random_states = []

# Avaliar modelos (naïve Bayes/kNN) 30 vezes, variando o random_state
for i in range(30):
    random_state = np.random.randint(0, 1000)  # Gerar um random_state aleatório
    random_states.append(random_state)

    # Dividir os dados entre treino e teste (proporção fixa)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # Normalizar dados
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # # Treinar o Naive Bayes
    # nb_model = GaussianNB()
    # nb_model.fit(X_train, y_train)

    # # Classificação e avaliação no conjunto de teste
    # y_pred_nb = nb_model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred_nb)
    # accuracies.append(accuracy)

    # Treinar o kNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Classificação e avaliação no conjunto de teste
    y_pred_knn = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_knn)
    accuracies.append(accuracy)


    # Exibir o desempenho a cada iteração
    print(f"Iteração {i+1}: Random State={random_state}, Acurácia={accuracy:.2f}")

# Plotar a variação das acurácias
plt.figure(figsize=(10,6))
plt.plot(range(1, 31), accuracies, marker='o', linestyle='--', color='b')
plt.title('Variação da Acurácia do Modelo em 30 Iterações com Random State Diferente')
plt.xlabel('Iteração')
plt.ylabel('Acurácia')
plt.xticks(range(1, 31))
plt.grid(True)
plt.show()

# Amplitude dos resultados
max(accuracies) - min(accuracies)
print(f"Amplitude das Acurácias: {max(accuracies) - min(accuracies):.2f}")

##################################################################
# Analisando o impacto do tamanho do conjunto de teste na avaliação de desempenho dos modelos

# Inicializar listas para armazenar resultados
variances = []
amplitudes = []
avg_accuracies = []

# Definir as proporções de conjunto de teste
test_sizes = np.arange(0.05, 0.70, 0.05)

# Loop para cada proporção de conjunto de teste
for test_size in test_sizes:
    accuracies = []

    # Repetir o experimento 30 vezes para cada tamanho de conjunto de teste
    for i in range(30):
        random_state = np.random.randint(0, 1000)

        # Dividir os dados com a proporção especificada para o conjunto de teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Normalizar dados
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # # Treinar o Naive Bayes
        # nb_model = GaussianNB()
        # nb_model.fit(X_train, y_train)

        # # Previsão e avaliação no conjunto de teste
        # y_pred_nb = nb_model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred_nb)
        # accuracies.append(accuracy)

        # Treinar o kNN
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)

        # Previsão e avaliação no conjunto de teste
        y_pred_knn = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_knn)
        accuracies.append(accuracy)

    # Calcular variância, amplitude e média dos desempenhos
    variance = np.var(accuracies)
    amplitude = np.max(accuracies) - np.min(accuracies)
    avg_accuracy = np.mean(accuracies)

    # Armazenar os resultados
    variances.append(variance)
    amplitudes.append(amplitude)
    avg_accuracies.append(avg_accuracy)

    # Exibir os resultados intermediários
    print(f"Tamanho do Conjunto de Teste: {test_size*100:.1f}%")
    print(f"   Média da Acurácia: {avg_accuracy:.3f}")
    print(f"   Variância: {variance:.5f}")
    print(f"   Amplitude (Máx - Mín): {amplitude:.3f}")
    print("")

# Gráfico: Variação da Acurácia com Diferentes Tamanhos de Conjunto de Teste
plt.figure(figsize=(10,6))
plt.plot(test_sizes * 100, avg_accuracies, marker='o', linestyle='--', color='b', label="Média das Acurácias")
plt.title('Média da Acurácia do KNN com Diferentes Tamanhos de Conjunto de Teste')
plt.xlabel('Tamanho do Conjunto de Teste (%)')
plt.ylabel('Média da Acurácia')
plt.grid(True)
plt.legend()
plt.show()

# Gráfico: Variância vs Tamanho do Conjunto de Teste
plt.figure(figsize=(10,6))
plt.scatter(test_sizes * 100, variances, color='r', label="Variância do Desempenho")
plt.title('Variância do Desempenho vs Tamanho do Conjunto de Teste')
plt.xlabel('Tamanho do Conjunto de Teste (%)')
plt.ylabel('Variância do Desempenho')
plt.grid(True)
plt.legend()
plt.show()

# Gráfico: Amplitude vs Tamanho do Conjunto de Teste
plt.figure(figsize=(10,6))
plt.scatter(test_sizes * 100, amplitudes, color='g', label="Amplitude (Máx - Mín)")
plt.title('Amplitude do Desempenho vs Tamanho do Conjunto de Teste')
plt.xlabel('Tamanho do Conjunto de Teste (%)')
plt.ylabel('Amplitude do Desempenho')
plt.grid(True)
plt.legend()
plt.show()


