import numpy as np
import pandas as pd

colunas = ['x', 'y']

# Carregue os dados do arquivo CSV usando pandas
df = pd.read_csv('artificial1d.csv', names=colunas)

# Separe os dados em x e y
x = df['x']
y = df['y']

# Calcule as médias de x e y
media_x = np.mean(x)
media_y = np.mean(y)

# Calcule os parâmetros do modelo de regressão linear
coef_angular = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x) ** 2)
coef_linear = media_y - coef_angular * media_x

# Calcule as previsões do modelo
y_pred = coef_angular * x + coef_linear

# Calcule o MSE
mse = np.mean((y - y_pred) ** 2)

# Imprima os parâmetros do modelo e o MSE
print(f"Coeficiente Angular: {coef_angular:.2f}")
print(f"Coeficiente Linear: {coef_linear:.2f}")
print(f"MSE: {mse:.2f}")