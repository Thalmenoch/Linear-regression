import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def MSE(y, y_previsto):
    return np.mean((y - y_previsto) ** 2)

def re2(y_teste, y_pred):
    return  1 - np.sum((y_teste - y_pred) ** 2) / np.sum((y_teste - np.mean(y_teste)) ** 2)

def regularizacao(x_treino_poly, y_treino):
    # Defina o parâmetro de regularização (alpha)
    alpha = 1.0  # Ajuste conforme necessário

    # Calcule os coeficientes regularizados usando a fórmula da regressão Ridge (L2)
    I = np.identity(x_treino_poly.shape[1])

    # Calcule os coeficientes do modelo de regressão polinomial usando a fórmula dos mínimos quadrados
    coeficientes = np.linalg.inv(x_treino_poly.T @ x_treino_poly + alpha * I) @ x_treino_poly.T @ y_treino

    return coeficientes

def treinamento_do_modelo(x_treino_norm, x_teste_norm, y_treino, y_teste):
    # Inicialize listas para armazenar métricas de desempenho
    lista_mse = []
    lista_r2 = []

    # Loop através de diferentes ordens de polinômio
    for ordem in range(1, 12):
        # Crie a matriz de atributos polinomiais para o conjunto de treinamento e teste
        X_treino_poly = np.column_stack([x_treino_norm ** i for i in range(1, ordem + 1)])
        X_teste_poly = np.column_stack([x_teste_norm ** i for i in range(1, ordem + 1)])
        
        coeficientes = regularizacao(X_treino_poly, y_treino)
        
        # Faça previsões no conjunto de teste
        y_pred = X_teste_poly @ coeficientes

        # Calcule as métricas de desempenho (por exemplo, MSE ou R²)
        mse = MSE(y_teste, y_pred)
        lista_mse.append(mse)

        r2 = re2(y_teste, y_pred)
        lista_r2.append(r2)

        # Imprima as métricas de desempenho
        print(f"Ordem do Polinômio: {ordem}")
        print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
        print(f"Coeficiente de Determinação (R²): {r2:.2f}")
        print("=" * 40)

    return lista_mse, lista_r2

if __name__ == '__main__':

    # Carregue o conjunto de dados a partir do arquivo CSV
    data = pd.read_csv('regressao_polinomial/boston.csv')

    # Separe os atributos (13 primeiras colunas) e as saídas (última coluna)
    x = data.iloc[:, :-1]  # Atributos
    y = data.iloc[:, -1]   # Saídas

    # Divida o conjunto de dados em treino (80%) e teste (20%)
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    x_treino_norm = scaler.fit_transform(x_treino)
    x_teste_norm = scaler.transform(x_teste)
    # Suponha que você já normalizou seus atributos (x_treino_norm, x_teste_norm) e tem seus valores de saída (y_treino, y_teste) disponíveis.

    lista_mse, lista_r2 = treinamento_do_modelo(x_treino_norm, x_teste_norm, y_treino, y_teste)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 12), lista_mse, marker='o')
    plt.title('Erro Quadrático Médio (MSE)')
    plt.xlabel('Ordem do Polinômio')
    plt.ylabel('MSE')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 12), lista_r2, marker='o')
    plt.title('Coeficiente de Determinação (R²)')
    plt.xlabel('Ordem do Polinômio')
    plt.ylabel('R²')

    plt.tight_layout()
    plt.show()