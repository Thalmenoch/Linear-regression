import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para calcular o erro quadrático médio (MSE)
def calcular_mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# Gradiente Descendente
def gradiente_descendente(x, y, taxa_aprendizado, num_iteracoes):
    # Inicialize os parâmetros do modelo
    coef_angular = 0
    intercepto = 0
    m = len(x)
    
    # Lista para armazenar o histórico do MSE
    mse_hist = []
    
    for _ in range(num_iteracoes):
        # Calcule as previsões do modelo
        y_pred = coef_angular * x + intercepto
        
        # Calcule o gradiente dos parâmetros
        gradiente_coef = (-2/m) * np.sum(x * (y - y_pred))
        gradiente_intercepto = (-2/m) * np.sum(y - y_pred)
        
        # Atualize os parâmetros usando o gradiente
        coef_angular -= taxa_aprendizado * gradiente_coef
        intercepto -= taxa_aprendizado * gradiente_intercepto
        
        # Calcule o MSE atual e o armazene
        mse_atual = calcular_mse(y, y_pred)
        mse_hist.append(mse_atual)
    
    return coef_angular, intercepto, mse_hist

colunas = ['x', 'y']

df = pd.read_csv('artificial1d.csv', names=colunas) 

x = df['x']
y = df['y']

# Médias
media_x = np.mean(x)
media_y = np.mean(y)

# Defina hiperparâmetros do Gradiente Descendente
taxa_aprendizado = 0.01
num_iteracoes = 1000

# Execute o Gradiente Descendente
coef_angular, intercepto, mse_hist = gradiente_descendente(x, y, taxa_aprendizado, num_iteracoes)

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

# Plote a reta resultante sobre os dados
y_pred = coef_angular * x + intercepto
plt.scatter(x, y, label='Dados Originais')
plt.plot(x, y_pred, color='red', label='Regressão Linear')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Regressão Linear com Gradiente Descendente')
plt.show()