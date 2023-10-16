import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Função para calcular a acurácia
def calcular_acuracia(y_verdadeiro, y_predito):
    return np.mean(y_verdadeiro == y_predito)

# Função para calcular a revocação
def calcular_revocacao(y_verdadeiro, y_predito):
    tp = np.sum((y_verdadeiro == 1) & (y_predito == 1))
    fn = np.sum((y_verdadeiro == 1) & (y_predito == 0))
    return tp / (tp + fn)

# Função para calcular a precisão
def calcular_precisao(y_verdadeiro, y_predito):
    tp = np.sum((y_verdadeiro == 1) & (y_predito == 1))
    fp = np.sum((y_verdadeiro == 0) & (y_predito == 1))
    return tp / (tp + fp)

# Função para calcular o F1-Score
def calcular_f1_score(y_verdadeiro, y_predito):
    precisao = calcular_precisao(y_verdadeiro, y_predito)
    revocacao = calcular_revocacao(y_verdadeiro, y_predito)
    return 2 * (precisao * revocacao) / (precisao + revocacao)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calcular_custo(theta, x, y):
    m = len(y)
    h = sigmoid(np.dot(x, theta))
    custo = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
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
        print(f'Época {epoch + 1}/{num_epochs}, Custo: {custo}')

    return theta

if __name__ == '__main__':
    # Substitua 'seuarquivo.csv' pelo nome do seu arquivo CSV
    arquivo_csv = 'seuarquivo.csv'

    # Use o método read_csv para ler o arquivo CSV
    df = pd.read_csv(arquivo_csv)

    # Separar as colunas em atributos e saída
    atributos = df.iloc[:, :-1]  # Todas as colunas, exceto a última
    saida = df.iloc[:, -1]  # Última coluna

    # Converter os DataFrames para arrays NumPy
    X = atributos.values
    y = saida.values

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calcular métricas
    acuracia = accuracy_score(y_teste, y_predito)
    revocacao = recall_score(y_teste, y_predito)
    precisao = precision_score(y_teste, y_predito)
    f1 = f1_score(y_teste, y_predito)

    # Matriz de confusão
    cm = confusion_matrix(y_teste, y_predito)

    # Apresentar as métricas
    print(f'Acurácia: {acuracia:.2f}')
    print(f'Revocação: {revocacao:.2f}')
    print(f'Precisão: {precisao:.2f}')
    print(f'F1-Score: {f1:.2f}')

    # Apresentar a matriz de confusão
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    classes = ['Classe Negativa', 'Classe Positiva']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.show()