import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

if __name__ == '__main__':

    colunas = ['x', 'y']

    df = pd.read_csv('artificial1d.csv', names=colunas) 

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