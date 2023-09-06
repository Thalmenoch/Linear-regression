import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cálculo do MSE
def MSE(y, y_previsto):
    return np.mean((y - y_previsto) ** 2)

#Cálculo dos coeficientes
def coef_linear_angular(x, y, media_x, media_y):
    coef_angular = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x) ** 2)
    coef_linear = media_y - coef_angular * media_x

    return coef_angular, coef_linear 

if __name__ == '__main__':

    colunas = ['x', 'y']

    df = pd.read_csv('regressão_linear_simples/artificial1d.csv', names=colunas) 

    x = df['x']
    y = df['y']

    # Médias
    media_x = np.mean(x)
    media_y = np.mean(y)

    #Parâmetros do modelo
    coef_angular, coef_linear = coef_linear_angular(x, y, media_x, media_y)

    # Calcule as previsões do modelo
    y_previsto = coef_angular * x + coef_linear
    
    #variável que recebe a função MSE
    mse = MSE(y, y_previsto)

    plt.scatter(x, y, label='Dados Recebidos')
    plt.plot(x, y_previsto, color='red', label=f'Regressão Linear (MSE={mse:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Regressão Linear')
    plt.show()

    print(f"Coeficiente Angular: {coef_angular:.2f}")
    print(f"Coeficiente Linear: {coef_linear:.2f}")
    print(f"MSE: {mse:.2f}")