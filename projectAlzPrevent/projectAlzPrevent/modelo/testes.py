# Importar bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
import io


# Função para calcular a probabilidade posterior de desenvolver Alzheimer para cada exposição
def calcular_probabilidades(exposicoes, probabilidade_alzheimer):
    probabilidades = []
    for index, exposicao in exposicoes.iterrows():
        pval = exposicao['pval']
        or_uci95 = exposicao['or_uci95']
        or_lci95 = exposicao['or_lci95']
        or_uci95 = exposicao['or_uci95']
        od_r = exposicao['or']

        # Calcular a probabilidade posterior usando o Teorema de Bayes
        probabilidade_posterior = (probabilidade_alzheimer * od_r) / ((probabilidade_alzheimer * od_r) + ((1 - probabilidade_alzheimer) * (1 - od_r)))

        probabilidades.append(probabilidade_posterior)
    return probabilidades

# Carregar os dados

from google.colab import files
uploaded = files.upload()
#dados = pd.read_csv('seu_arquivo.csv')  # Substitua 'seu_arquivo.csv' pelo caminho do arquivo CSV
dados = pd.read_csv(io.BytesIO(uploaded['banco_expo2.csv']))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

# Definir a probabilidade inicial de desenvolver Alzheimer (prior)
probabilidade_alzheimer = 0.1  # Por exemplo, 10%

# Filtrar apenas as colunas relevantes para cálculo de probabilidades
exposicoes = dados[['exposure', 'pval', 'or', 'or_lci95', 'or_uci95']]

# Calcular as probabilidades para cada exposição
probabilidades = calcular_probabilidades(exposicoes, probabilidade_alzheimer)

print('Probabilidaes para cada exposição:', probabilidades, '\n')


import numpy as np

# Gerar valores aleatórios para as exposições para 100 indivíduos
np.random.seed(42)  # Define a semente aleatória para reproduzibilidade
valores_exposicoes = np.random.rand(100, exposicoes.shape[0])  # 100 indivíduos, cada um com valores para cada exposição

print("Exposições:", exposicoes,'\n')
print("Valores das exposições:", valores_exposicoes,'\n')

print("Shape dos DataFrame de Individuos:", valores_exposicoes.shape, '\n')

# Calcular a probabilidade de desenvolvimento de Alzheimer para cada indivíduo
#probabilidades_individuos = np.dot(valores_exposicoes, np.array(probabilidades))

from sklearn.linear_model import LogisticRegression

# Verificar se o número de exposições para ajustar o modelo é igual ao número de exposições usadas para calcular as probabilidades
if len(valores_exposicoes[0]) != len(probabilidades):
    raise ValueError("O número de exposições para ajustar o modelo não é igual ao número de exposições usadas para calcular as probabilidades.")


# Criar e ajustar o modelo de Regressão Logística
modelo_logistico = LogisticRegression()
modelo_logistico.fit(valores_exposicoes, np.array(probabilidades))

# Calcular as probabilidades individuais de desenvolver Alzheimer para cada indivíduo
probabilidades_individuos = modelo_logistico.predict_proba(valores_exposicoes)[:, 1]  # Seleciona as probabilidades de classe positiva

print(probabilidades_individuos)

# Definir um limite de decisão
limiar = 0.5  # Limiar de 50%

# Classificar cada indivíduo com base na probabilidade calculada
classificacoes = np.where(probabilidades_individuos > limiar, 1, 0)

## Criar o dataframe com as exposições, a probabilidade calculada e a classificação
nomes_individuos = ['Individuo{}'.format(i+1) for i in range(100)]
df_individuos = pd.DataFrame(valores_exposicoes, columns=exposicoes['exposure'], index=nomes_individuos)
df_individuos['probabilidade'] = probabilidades_individuos
df_individuos['classificacao'] = classificacoes

# Exportar o DataFrame para um arquivo CSV
print(df_individuos.head())
df_individuos.to_csv('df_individuos.csv')

# Criar um DataFrame com as probabilidades e as exposições
df_probabilidades = pd.DataFrame({'exposure': exposicoes['exposure'], 'probabilidade': probabilidades})

print('DataFrame com as probabilidades e as exposições:\n', df_probabilidades, '\n')

# Definir um limite de decisão
limiar = 0.5  # Limiar de 50%

# Classificar os indivíduos com base nas probabilidades
df_probabilidades['classificacao'] = df_probabilidades['probabilidade'].apply(lambda x: 1 if x > limiar else 0)

print('DataFrame com as probabilidades, exposições e classificacao:\n', df_probabilidades)

# Dividir os dados em variáveis de entrada (exposições) e saída (classificação)
X = df_probabilidades['probabilidade'].values.reshape(-1, 1)  # Usaremos apenas as probabilidades como variável de entrada
y = df_probabilidades['classificacao']

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de Regressão Logística com validação cruzada
modelo = LogisticRegression()
scores = cross_val_score(modelo, X_train, y_train, cv=5)  # Validacão cruzada com 5 folds
modelo.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_test)

# Avaliar o desempenho do modelo
acuracia = accuracy_score(y_test, previsoes)
precisao = precision_score(y_test, previsoes)
recall = recall_score(y_test, previsoes)
f1 = f1_score(y_test, previsoes)
roc_auc = roc_auc_score(y_test, previsoes)

print("Acurácia do modelo:", acuracia)
print("Precisão do modelo:", precisao)
print("Recall do modelo:", recall)
print("F1-score do modelo:", f1)
print("Área sob a curva ROC (AUC) do modelo:", roc_auc)
print("\nRelatório de Classificação:")
print(classification_report(y_test, previsoes))