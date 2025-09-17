## Carregando as bibliotecas necessárias
# %matplotlib inline
import pandas as pd             # para análise de dados
import matplotlib.pyplot as plt # para visualização de informações
import seaborn as sns           # para visualização de informações
import numpy as np              # para operações com arrays multidimensionais
from sklearn.svm import SVC  ## para treinar um SVM
from sklearn.model_selection import train_test_split # para divisão de dados
from sklearn.metrics import confusion_matrix, recall_score, precision_score,accuracy_score,ConfusionMatrixDisplay ## para avaliação dos modelos
sns.set()

df = pd.read_table("https://drive.google.com/uc?export=view&id=11YsVJck74_gyADzGJSU9Uwn8cDq_l3BD",sep=";")
df.head()  # para visualizar apenas as 5 primeiras linhas

## Características gerais do dataset
print("O conjunto de dados possui {} linhas e {} colunas".format(df.shape[0], df.shape[1]))

## Distribuição do atributo alvo, 'quality'
sns.countplot(x='quality', data=df)

## Imprimindo o valor exato de número de instâncias por nota
print(df.groupby('quality').size())

df.info()

## Para analisar valores faltantes, quando codificados como NaN, podemos usar o
## comando abaixo
df.isnull().sum()

## Criando vetor com nome dos atributos
features_names = df.columns.drop(['quality'])

## Gerar um gráfico para cada variável numérica com a distribuição
## de frequência. Avaliar a distribuição geral ou, opcionalmente,
## a distribuição por classe (classificação do vinho)

## Distribuição geral
def dist_plot(df,columns):
    plt.figure(figsize=(16, 10))
    for indx, var  in enumerate(columns):
        plt.subplot(4, 3, indx+1)
        g = sns.histplot(x=var, data=df)
    plt.tight_layout()

## Distribuição por classe
def dist_plot_perclass(df,columns,label):
    plt.figure(figsize=(16, 10))
    for indx, var  in enumerate(columns):
        plt.subplot(4, 3, indx+1)
        sns.color_palette("pastel")
        g = sns.histplot(x=var, data=df, hue=label,palette='muted')
    plt.tight_layout()


dist_plot(df, features_names)
#dist_plot_perclass(df, features_names, 'quality')

df.drop(['quality'],axis=1).plot(figsize=(15,7))

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap = 'coolwarm')
plt.show()

feature_sel = "density" #@param {type:"string"}

plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="quality",y=feature_sel,color='r')

df['quality'] = df['quality'].replace([3, 4, 5, 6], 0)
df['quality'] = df['quality'].replace([7, 8, 9], 1)
sns.countplot(x='quality', data=df)

## Separa o dataset em duas variáveis: os atributos/entradas (X) e a classe/saída (y)
X = df.drop(['quality'], axis=1)
y = df['quality'].values

## Definindo as proporções de treino, validação e teste.
train_ratio = 0.70
test_ratio = 0.15
validation_ratio = 0.15

## Fazendo a primeira divisão, para separar um conjunto de teste dos demais.
## Assuma X_train e y_train para os dados de treinamento e X_test e y_test para os de teste
## Dica: configure o random_state para facilitar reprodutibilidade dos experimentos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,stratify=y,random_state=42)

## Fazendo a segunda divisão, para gerar o conjunto de treino e validação a partir
## do conjunto de 'treinamento' da divisão anterior
## Assuma X_train e y_train para os dados de treinamento e X_valid e y_valid para os de teste
## Dica: configure o random_state para facilitar reprodutibilidade dos experimentos
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio),stratify=y_train,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

from sklearn.preprocessing import MinMaxScaler
## O MinMaxScaler transformará os dados para que fiquem no intervalo [0,1]
scaler = MinMaxScaler() #StandardScaler()

## Iniciar a normalização dos dados. Primeiro fazer um 'fit' do scaler nos
## dados de treino. Esta etapa visa "aprender" os parâmetros para normalização.
## No caso do MinMaxScales, são os valores mínimos e máximos de cada atributo
scaler.fit(X_train)

## Aplicar a normalização nos três conjuntos de dados:
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

df_train_norm =  pd.DataFrame(X_train,columns=features_names)
df_ytrain =  pd.DataFrame(y_train,columns=['quality'],dtype=int)
df_train_norm =  pd.concat([df_train_norm,df_ytrain], axis=1)
ave_values = df_train_norm.groupby("quality").median()
ave_values.plot(kind="bar",figsize=(15,7))

## Definindo um array para armazenar o desempenho de cada modelo treinado e avaliado
perf_valid = []

## Definindo valores de C a serem testados
param_grid_C = [0.1, 1, 5, 10, 50, 100] #valores para C, termo de regularização

# Treinando e avaliado os modelos com cada valor de hiperparâmetro especificado
for ii in range(len(param_grid_C)):
    clf = SVC(kernel='linear',C=param_grid_C[ii],random_state=42, class_weight='balanced') ##class_weight minimiza o efeito de termos mais exemplos para vinhos medíocres/ruins
    clf.fit(X_train, y_train)
    pred_i = clf.predict(X_valid)
    perf_valid.append([param_grid_C[ii],accuracy_score(y_valid, pred_i),recall_score(y_valid, pred_i),precision_score(y_valid, pred_i)])

perf_df = pd.DataFrame(perf_valid, columns=['C','accuracy','recall','precision'])
perf_df

plt.figure(figsize=(12, 6))
## Transforma o dataframe para facilitar plotar todas as métricas na mesma figura
perf_df_melt = pd.melt(perf_df, id_vars=['C'], value_vars=['accuracy','recall','precision'])
sns.lineplot(data=perf_df_melt,x='C',y='value',hue='variable',palette='muted',marker='o')

from sklearn.model_selection import PredefinedSplit, GridSearchCV ## para auxiliar na otimização de hiperparâmetros

# Cria lista com os dados de treinamento com índice -1 e dados de validação com índice 0
# Concatena os dados de treino e validação com as partições pré-definidas
split_index = [-1]*len(X_train) + [0]*len(X_valid)
X_gridSearch = np.concatenate((X_train, X_valid), axis=0)
y_gridSearch = np.concatenate((y_train, y_valid), axis=0)
pds = PredefinedSplit(test_fold = split_index)

## Define métricas de desempenho a serem estimadas
scoring = {'Accuracy':'accuracy', 'Precision': 'precision', 'Recall':'recall'}

## Define o algoritmo base da otimização de hiperparâmetros
estimator = SVC(kernel='linear',class_weight='balanced')

## Define a grid de hiperparâmetros a serem testados
param_grid = {'C': [0.1, 1, 5, 10, 50]}
#param_grid = {'C': [0.1, 1, 5, 10, 50, 100]}#, 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
#param_grid = {'C': [0.1, 1, 5, 10, 50, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}


## Aplica GridSearch com as partições de treino/validação pré-definidas
gridS = GridSearchCV(estimator = estimator,
                   cv=pds,
                   param_grid=param_grid,
                   scoring=scoring,
                   refit='Accuracy', ##métrica a ser utilizada para definir o melhor modelo, retreinando-o com toda a base
                   return_train_score=True)
gridS.fit(X_gridSearch, y_gridSearch)
print('Desempenho máximo obtido com: {}'.format(gridS.best_params_))

print(gridS.cv_results_)

## O código desta célula cria um gráfico de variação de desempenho de acordo com
## o valor do hiperparâmetro C.

results = gridS.cv_results_

plt.figure(figsize=(10, 7))
plt.title("Resultados do GridSearchCV",
      fontsize=16)

plt.xlabel("Hyperparameter") ##nome do parâmetro a ser analisado
plt.ylabel("Performance")

ax = plt.gca()

## Criar um numpy array para os resultados do hiperparâmetro a ser analisado.
## O hiperparâmetro C está identificado no objeto retornado pelo gridSearchCV
## como param_C
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k', 'b', 'r']):
    for sample, style in (('train', '--'), ('test', '-')):
       sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
       sample_score_std = results['std_%s_%s' % (sample, scorer)]
       ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
       ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    ## Plota uma linha vertical para o valor de hiperparâmetro que maximiza a métrica de desempenho
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
        linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
    ## Anota o valor do melhor score
    ax.annotate("%0.3f" % best_score,
            (X_axis[best_index], best_score + 0.008))

plt.legend(loc="best")
plt.grid(False)
plt.show()

y_pred_svmLinear = gridS.predict(X_test) ##predição usando SVM com a melhor configuração de hiperparâmetros encontrada

cm = confusion_matrix(y_test, y_pred_svmLinear,labels=gridS.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gridS.classes_)
disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
plt.grid(False)
plt.show()

print('Acurácia: {}'.format(round(accuracy_score(y_test, y_pred_svmLinear),3)))
print('Recall: {}'.format(round(recall_score(y_test, y_pred_svmLinear,pos_label=1),3)))
print('Precisão: {}'.format(round(precision_score(y_test, y_pred_svmLinear,pos_label=1),3)))

from sklearn.model_selection import PredefinedSplit, GridSearchCV ## para auxiliar na otimização de hiperparâmetros


## Cria lista com os dados de treinamento com índice -1 e dados de validação com índice 0
## Concatena os dados de treino e validação com as partições pré-definidas
split_index = [-1]*len(X_train) + [0]*len(X_valid)
X_gridSearch = np.concatenate((X_train, X_valid), axis=0)
y_gridSearch = np.concatenate((y_train, y_valid), axis=0)
pds = PredefinedSplit(test_fold = split_index)

# ## Define métricas de desempenho a serem estimadas
scoring = {'Accuracy':'accuracy', 'Precision': 'precision', 'Recall':'recall'}

## Define o algoritmo base da otimização de hiperparâmetros
estimator = SVC(class_weight='balanced')

## Define a grid de hiperparâmetros a serem testados
param_grid = {'C': [0.1, 1, 5, 10, 50, 100], 'kernel': ['rbf', 'poly', 'sigmoid']}#,'gamma': [1,0.1,0.01,0.001]} ## gamma foi removido para reduzir tempo de execução

## Aplica GridSearch com as partições de treino/validação pré-definidas
gridS2 = GridSearchCV(estimator = estimator,
                   cv=pds,
                   param_grid=param_grid,
                   scoring=scoring,
                   refit='Accuracy', ##métrica a ser utilizada para definir o melhor modelo, retreinando-o com toda a base
                   return_train_score=True)
gridS2.fit(X_gridSearch, y_gridSearch)
print('Desempenho máximo obtido com: {}'.format(gridS2.best_params_))

results = gridS2.cv_results_

plt.figure(figsize=(10, 7))
plt.title("Resultados do GridSearchCV",
      fontsize=16)

plt.xlabel("Hyperparameter") ##nome do parâmetro a ser analisado
plt.ylabel("Performance")

ax = plt.gca()

## Criar um numpy array para os resultados do hiperparâmetro a ser analisado
X_axis = np.array(results['param_kernel'].data)#, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k', 'b', 'r']):
    for sample, style in (('train', '--'), ('test', '-')):
       sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
       sample_score_std = results['std_%s_%s' % (sample, scorer)]
       ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
       ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    ## Plota uma linha vertical para o valor de hiperparâmetro que maximiza a métrica de desempenho
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
        linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
    ## Anota o valor do melhor score
    ax.annotate("%0.3f" % best_score,
            (X_axis[best_index], best_score + 0.008))

plt.legend(loc="best")
plt.grid(False)
plt.show()

