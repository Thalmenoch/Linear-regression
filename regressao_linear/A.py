import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cálculo do MSE
def MSE(y, y_previsto):
    return np.mean((y - y_previsto) ** 2)

#Cálculo dos coeficientes da OLS
def OLS(x, y, media_x, media_y):
    coef_angular = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x) ** 2)
    vies = media_y - coef_angular * media_x

    return coef_angular, vies 

if __name__ == '__main__':
    #Nomeando as colunas
    colunas = ['a', 'b']

    df = pd.read_csv('regressão_linear_simples/artificial1d.csv', names=colunas) 

    x = df['a']
    y = df['b']

    x = (x - x.mean()) / x.std()

    #Médias
    media_x = np.mean(x)
    media_y = np.mean(y)

    #Parâmetros do modelo
    coef_angular, vies = OLS(x, y, media_x, media_y)

    #Cálculo das previsões do modelo
    y_previsto = coef_angular * x + vies
    
    #Variável que recebe a função MSE
    mse = MSE(y, y_previsto)
    
    #Plotando a reta de regressão sobre os dados
    plt.scatter(x, y, label='Dados Recebidos')
    plt.plot(x, y_previsto, color='red', label='Regressão Linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Regressão Linear com OLS')
    plt.show()

    print(f"Coeficiente Angular: {coef_angular:.2f}")
    print(f"Viés: {vies:.2f}")
    print(f"MSE: {mse:.2f}")