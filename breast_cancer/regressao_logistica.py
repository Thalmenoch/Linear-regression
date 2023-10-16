import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from metodos import *

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calcular_custo(theta, x, y):
    m = len(y)
    h = sigmoid(np.dot(x, theta))
    # custo = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    epsilon = 1e-10
    custo = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return custo

# Função de treinamento da regressão logística com Gradient Descent
def treinar_regressao_logistica(x, y, learning_rate, num_epochs):
    m, n = x.shape
    theta = np.zeros(n)

    for epoch in range(num_epochs):
        z = np.dot(x, theta)
        h = sigmoid(z)
        gradient = (1/m) * np.dot(x.T, (h - y))
        theta -= learning_rate * gradient

        custo = calcular_custo(theta, x, y)
        print(f'Época {epoch + 1}/{num_epochs}, Custo: {custo:.2f}')

    return theta

def prever(x, theta):
    probabilidade = sigmoid(np.dot(x, theta))
    return (probabilidade >= 0.5).astype(int)

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

    learning_rate = 0.01
    num_epochs = 100

    # Treinar a regressão logística no conjunto de treinamento
    theta_treinado = treinar_regressao_logistica(X_treino, y_treino, learning_rate, num_epochs)

    y_predito = prever(X_teste, theta_treinado)

    TP, FP, TN, FN = calcular_matriz_confusao(y_teste, y_predito)

    # Calcular métricas
    acuracia = calcular_acuracia(y_teste, y_predito)
    revocacao = calcular_revocacao(y_teste, y_predito)
    precisao = calcular_precisao(y_teste, y_predito)
    f1 = calcular_f1_score(y_teste, y_predito)

    # Apresentar as métricas
    print(f'Acurácia: {acuracia:.2f}')
    print(f'Revocação: {revocacao:.2f}')
    print(f'Precisão: {precisao:.2f}')
    print(f'F1-Score: {f1:.2f}')
 
    plot_confusion_matrix(TP, FP, TN, FN)
    plot_metrics(acuracia, precisao, revocacao, f1)
    