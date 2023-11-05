import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from metodos import *

# Função para calcular os parâmetros do GDA
def calcular_parametros(x, y):
    classes = np.unique(y)
    parametros = []

    for classe in classes:
        x_classe = x[y == classe]
        media = np.mean(x_classe, axis=0)
        covariancia = np.cov(x_classe, rowvar=False)
        parametros.append((media, covariancia))

    return parametros

# Função para calcular a probabilidade da classe usando a distribuição gaussiana
def calcular_probabilidade_classe(x, media, covariancia):
    n = len(media)
    det_cov = np.linalg.det(covariancia)
    inv_cov = np.linalg.pinv(covariancia)
    x_minus_media = x - media

    exponent = -0.5 * np.dot(x_minus_media, np.dot(inv_cov, x_minus_media))
    probabilidade = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_cov)) * np.exp(exponent)

    return probabilidade

# Função para fazer previsões
def prever(x, parametros):
    previsoes = []
    for x in x:
        probabilidades_classe = []
        for media, covariancia in parametros:
            probabilidade = calcular_probabilidade_classe(x, media, covariancia)
            probabilidades_classe.append(probabilidade)
        classe_prevista = np.argmax(probabilidades_classe)
        previsoes.append(classe_prevista)
    return previsoes
    

if __name__ == '__main__':
    arquivo_csv = 'breast_cancer/breast.csv'
    df = pd.read_csv(arquivo_csv)

    # Separar as colunas em atributos e saída
    atributos = df.iloc[:, :-1]
    saida = df.iloc[:, -1]

    # Converter os DataFrames para arrays NumPy
    X = atributos.values
    y = saida.values

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo GDA e calcular os parâmetros
    parametros = calcular_parametros(X_treino, y_treino)

    # Fazer previsões com o modelo GDA
    previsoes = prever(X_teste, parametros)

    # Calcular a matriz de confusão
    TP, FP, TN, FN = calcular_matriz_confusao(y_teste, previsoes)

    # Calcular métricas de desempenho
    acuracia = calcular_acuracia(TP, FP, TN, FN)
    revocacao = calcular_revocacao(TP, FN)
    precisao = calcular_precisao(TP, FP)
    f1 = calcular_f1_score(TP, FN)

    # Apresentar as métricas
    print(f'Acurácia: {acuracia:.2f}')
    print(f'Revocação: {revocacao:.2f}')
    print(f'Precisão: {precisao:.2f}')
    print(f'F1-Score: {f1:.2f}')

    # Plotar a matriz de confusão e métricas
    plot_confusion_matrix(TP, FP, TN, FN)
    plot_metrics(acuracia, precisao, revocacao, f1)
    