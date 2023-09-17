import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from regressao_linear.B import MSE
from sklearn.preprocessing import MinMaxScaler

# Carregue o conjunto de dados a partir do arquivo CSV
data = pd.read_csv('regressao_polinomial/boston.csv')

# Separe os atributos (13 primeiras colunas) e as saídas (última coluna)
x = data.iloc[:, :-1]  # Atributos
y = data.iloc[:, -1]   # Saídas

# Divida o conjunto de dados em treino (80%) e teste (20%)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# O argumento "test_size" define a proporção do conjunto de teste (20% neste caso)
# O argumento "random_state" é usado para garantir a reprodutibilidade do resultado

# Agora você tem os conjuntos de treino e teste prontos para serem usados em sua análise ou modelo de machine learning.
# Inicialize listas para armazenar métricas de desempenho
mse_scores = []
r2_scores = []

# 1. Normalização Min-Max
scaler = MinMaxScaler()
x_treino_norm = scaler.fit_transform(x_treino)
x_teste_norm = scaler.transform(x_teste)

#@ significa multiplicação de matrizes
for ordem in range(1, 12):
    # Aplique a transformação polinomial aos atributos de treinamento e teste
    x_treino_poly = np.column_stack([x_treino_norm ** i for i in range(1, ordem + 1)])
    x_teste_poly = np.column_stack([x_teste_norm ** i for i in range(1, ordem + 1)])

    # Calcule os coeficientes do modelo de regressão polinomial usando a fórmula dos mínimos quadrados
    coeficientes = np.linalg.inv(x_treino_poly.T @ x_treino_poly) @ x_treino_poly.T @ y_treino

    # Faça previsões no conjunto de teste
    y_pred = x_teste_poly @ coeficientes

    # Desfaça a normalização Min-Max nos valores previstos
    y_min = y.min()
    y_max = y.max()
    y_pred_desnormalizado = y_pred * (y_max - y_min) + y_min

    # Calcule o erro quadrático médio (MSE)
    mse = MSE(y_teste, y_pred)
    mse_scores.append(mse)

    # Calcule o coeficiente de determinação (R²)
    r2 = 1 - np.sum((y_teste - y_pred) ** 2) / np.sum((y_teste - np.mean(y_teste)) ** 2)
    r2_scores.append(r2)

# Plote um gráfico das métricas em relação à ordem do polinômio
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 12), mse_scores, marker='o')
plt.title('Erro Quadrático Médio (MSE)')
plt.xlabel('Ordem do Polinômio')
plt.ylabel('MSE')

# Imprima as métricas de desempenho
for ordem, mse, r2 in zip(range(1, 12), mse_scores, r2_scores):
    print(f"Ordem do Polinômio: {ordem}")
    print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
    print(f"Coeficiente de Determinação (R²): {r2:.4f}")
    print("=" * 40)
    
plt.show()
