from sklearn.datasets import load_breast_cancer
import pprint
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # função do scikit-learn que implementa um holdout
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data  # matriz contendo os atributos
y = data.target  # vetor contendo a classe (0 para maligno e 1 para benigno) de cada instância
feature_names = data.feature_names  # nome de cada atributo
target_names = data.target_names  # nome de cada classe

print(f"Dimensões de X: {X.shape}\n")
print(f"Dimensões de y: {y.shape}\n")
print(f"Nomes dos atributos: {feature_names}\n")
print(f"Nomes das classes: {target_names}")

# #print header
# header = "ID," + ",".join(feature_names) + ",class"
# print(header)
# # print data
# for i in range(X.shape[0]):
#     row = f"{i+1}," + ",".join(f"{v:.6f}" for v in X[i]) + f",{y[i]}"
#     print(row) 
    



# n_malign = np.sum(y == 0)
# n_benign = np.sum(y == 1)

# print("Número de exemplos malignos: %d" % n_malign)
# print("Número de exemplos benignos: %d" % n_benign)



#parte3

def get_root_node(dt, feature_names):
    feature_idx = dt.tree_.feature[0]
    return feature_names[feature_idx]

n_repeats = 20
root_nodes = []

# variando o seed do holdout, geramos conjuntos de treino e teste um pouco diferentes a cada iteração
for split_random_state in range(0, n_repeats):
  # Holdout com 20% de dados de teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_random_state)

  # Treinamento da árvore usando os dados de treino
  dt = DecisionTreeClassifier(random_state=0)
  dt.fit(X_train, y_train)

  # Obtemos o atributo usado na raiz e o salvamos na lista
  root_node = get_root_node(dt, feature_names)
  root_nodes.append(root_node)

pprint.pprint(root_nodes)

#contagem dos atributos usados na raiz
from collections import Counter
root_node_counts = Counter(root_nodes)
pprint.pprint(root_node_counts)


# #Para visualizar a estrutura da árvore

# from sklearn import tree

# dot_data = tree.export_graphviz(dt,
#                                 out_file=None,
#                                 feature_names = feature_names,
#                                 class_names= target_names,
#                                 filled=True)

# ## Plotar a árvore de decisão no notebook
# graph = graphviz.Source(dot_data)
# graph

# ## Para salvar como png, descomente as linhas abaixo
# # graph.format = 'png'
# # graph.render('DecisionTree1',view = True)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Acurácia nos dados de teste: %.3f" % accuracy)


#questão 2
n_repeats = 20
accuracies = []

# variando o seed do holdout, geramos conjuntos de treino e teste um pouco diferentes a cada iteração
for split_random_state in range(0, n_repeats):
  # Holdout com 20% de dados de teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_random_state)

  # Nova instância da árvore de decisão
  dt = DecisionTreeClassifier(random_state=0)

  # Treine a árvore de decisão usando os dados de treino
  dt.fit(X_train, y_train)
  y_pred = dt.predict(X_test)

  # Calcule a acurácia nos dados de teste
  accuracy = accuracy_score(y_test, y_pred)
  print("Acurácia nos dados de teste: %.3f" % accuracy)
  # Salve a acurácia na lista
  accuracies.append(accuracy)


# Calcule a média, desvio padrão, máximo e mínimo das acurácias (pode usar numpy)
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
max_accuracy = np.max(accuracies)
min_accuracy = np.min(accuracies)
print(f"Média: {mean_accuracy:.3f}")
print(f"Desvio Padrão: {std_accuracy:.3f}")
print(f"Máximo: {max_accuracy:.3f}")
print(f"Mínimo: {min_accuracy:.3f}")
print(f"Acurácias: {accuracies}")
print(f"Nomes dos atributos: {feature_names}\n")
print(f"Nomes das classes: {target_names}")


# ### Análise de Instância individuais ###

# X_interesting = X[[40, 86, 297, 135, 73], :]
# y_interesting = y[[40, 86, 297, 135, 73]]

# # 1. Instancie uma nova árvore de decisão, dessa vez sem especificar o valor de random_state
# dt = DecisionTreeClassifier(max_depth=2)

# # 2. Separe o conjunto em treino e teste, dessa vez sem especificar o valor de random_state
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # 3. Treine a nova árvore usando o conjunto de treino
# dt.fit(X_train, y_train)

# # 4. Use a nova árvore treinada para obter predições para os valores de X_interesting acima.
# y_interesting_pred = dt.predict(X_interesting)

# # 5. Imprima os valores preditos e os valores reais (y_interesting) para comparação.
# print("Valores preditos: ", y_interesting_pred)
# print("Valores reais:   ", y_interesting)

# # 6. Acurácia nos dados de teste
# y_test_pred = dt.predict(X_test)
# accuracy = accuracy_score(y_test, y_test_pred)
# print("Acurácia nos dados de teste: %.3f" % accuracy)

# # 7. Visualizar a árvore
# from sklearn import tree
# dot_data = tree.export_graphviz(dt,
#                                 out_file=None,
#                                 feature_names = feature_names,
#                                 class_names= target_names,
#                                 filled=True)
# ## Plotar a árvore de decisão no notebook
# graph = graphviz.Source(dot_data)
# graph
# ## Para salvar como png, descomente as linhas abaixo
# graph.format = 'png'
# graph.render('DecisionTree2',view = True)

