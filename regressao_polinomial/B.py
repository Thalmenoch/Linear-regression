import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregue o conjunto de dados a partir do arquivo CSV
data = pd.read_csv('regressao_polinomial/boston.csv')

# Separe os atributos (13 primeiras colunas) e as saídas (última coluna)
X = data.iloc[:, :-1]  # Atributos
y = data.iloc[:, -1]   # Saídas

# Divida o conjunto de dados em treino (80%) e teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialize listas para armazenar métricas de desempenho
mse_scores = []
r2_scores = []

# Loop através de diferentes ordens de polinômio
for ordem in range(1, 12):
    # Transforme os atributos para a ordem atual do polinômio
    poly = PolynomialFeatures(degree=ordem)
    X_treino_poly = poly.fit_transform(X_treino)
    X_teste_poly = poly.transform(X_teste)
    
    # Treine um modelo de regressão linear nos atributos transformados
    modelo = LinearRegression()
    modelo.fit(X_treino_poly, y_treino)
    
    # Faça previsões no conjunto de teste
    y_pred = modelo.predict(X_teste_poly)
    
    # Calcule o erro quadrático médio (MSE)
    mse = mean_squared_error(y_teste, y_pred)
    mse_scores.append(mse)
    
    # Calcule o coeficiente de determinação (R²)
    r2 = r2_score(y_teste, y_pred)
    r2_scores.append(r2)

# Plote um gráfico das métricas em relação à ordem do polinômio
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 12), mse_scores, marker='o')
plt.title('Erro Quadrático Médio (MSE)')
plt.xlabel('Ordem do Polinômio')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(range(1, 12), r2_scores, marker='o')
plt.title('Coeficiente de Determinação (R²)')
plt.xlabel('Ordem do Polinômio')
plt.ylabel('R²')

plt.tight_layout()
plt.show()
