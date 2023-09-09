import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colunas = ['x', 'y']

dados = pd.read_csv('regressão_linear_simples/artificial1d.csv', names=colunas) 

# Separe os dados em x e y
x = dados['x']
y = dados['y']

# Normalize as características (opcional, mas pode ajudar a convergência)
x = (x - x.mean()) / x.std()

# Defina hiperparâmetros do Gradiente Descendente 
taxa_aprendizado = 0.01
num_iteracoes = 1000

# Inicialize os parâmetros do modelo (coeficiente angular e intercepto)
coef_angular = 0
intercepto = 0

# Lista para armazenar o histórico do MSE
mse_hist = []

for _ in range(num_iteracoes):
    # Calcule as previsões do modelo
    y_pred = coef_angular * x + intercepto

    # Calcule o erro
    erro = y_pred - y

    # Atualize os parâmetros usando o gradiente
    gradiente_coef = (2 / len(x)) * np.sum(erro * x)
    gradiente_intercepto = (2 / len(x)) * np.sum(erro)

    coef_angular -= taxa_aprendizado * gradiente_coef
    intercepto -= taxa_aprendizado * gradiente_intercepto

    # Calcule o MSE atual e o armazene
    mse_atual = np.mean(erro**2)
    mse_hist.append(mse_atual)

# Imprima os parâmetros do modelo e o MSE
print(f"Coeficiente Angular: {coef_angular:.2f}")
print(f"Intercepto: {intercepto:.2f}")
print(f"Erro Quadrático Médio (MSE): {mse_hist[-1]:.2f}")

# Plote a curva de aprendizagem (MSE vs. Iterações)
plt.plot(range(1, num_iteracoes + 1), mse_hist)
plt.xlabel('Número de Iterações')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('Curva de Aprendizagem do Gradiente Descendente')
plt.show()

# Plote a reta de regressão sobre os dados
plt.scatter(x, y, label='Dados Originais')
plt.plot(x, coef_angular * x + intercepto, color='red', label='Regressão Linear (GD)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Regressão Linear com Gradiente Descendente')
plt.show()