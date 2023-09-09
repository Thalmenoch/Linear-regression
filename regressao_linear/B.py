import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Cálculo do MSE
def MSE(y, y_previsto):
    return np.mean((y - y_previsto) ** 2)

#Gradiente Descendente
def gradiente_descendente(x, y, taxa_aprendizado, nmr_iteracoes):
    #Inicialização dos parâmetros do modelo
    coef_angular = 0
    vies = 0
    norm = len(x) #Normaliza o gradiente pela quantidade de pontos de dados
    
    #Lista para armazenar os valores do MSE
    lista_mse = []
    
    for _ in range(nmr_iteracoes):
        #Cálculando as previsões do modelo
        y_previsto = coef_angular * x + vies
        
        #Cálculando os gradientes dos parâmetros
        gradiente_coef_angular = (-2/norm) * np.sum(x * (y - y_previsto))
        gradiente_vies = (-2/norm) * np.sum(y - y_previsto)
        
        #Atualizando os parâmetros usando o gradiente
        coef_angular -= taxa_aprendizado * gradiente_coef_angular
        vies -= taxa_aprendizado * gradiente_vies
        
        #Cálculo do MSE atual e o armazenenamento dele na lista
        mse_atual = MSE(y, y_previsto)
        lista_mse.append(mse_atual)
    
    return coef_angular, vies, lista_mse


if __name__ == '__main__':
    #Nomeando as colunas
    colunas = ['a', 'b']

    df = pd.read_csv('regressão_linear_simples/artificial1d.csv', names=colunas) 

    x = df['a']
    y = df['b']

    #Normalização das características
    x = (x - x.mean()) / x.std()

    #Hiperparâmetros do Gradiente Descendente
    taxa_aprendizado = 0.02
    # nmr_iteracoes = 500
    nmr_iteracoes = 60

    #Variáveis que recebem o Gradiente Descendente
    coef_angular, vies, lista_mse = gradiente_descendente(x, y, taxa_aprendizado, nmr_iteracoes)

    #Print dos parâmetros do modelo 
    print(f"Coeficiente Angular: {coef_angular:.2f}")
    print(f"Viés: {vies:.2f}")
    print(f"MSE: {lista_mse[-1]:.2f}")

    #Plotando a curva de aprendizagem 
    plt.plot(range(1, nmr_iteracoes + 1), lista_mse)
    plt.xlabel('Número de Iterações')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.title('Curva de Aprendizagem do Gradiente Descendente')
    plt.show()

    #Plotando a reta de regressão sobre os dados
    y_previsto = coef_angular * x + vies
    plt.scatter(x, y, label='Dados Recebidos')
    plt.plot(x, y_previsto, color='red', label='Regressão Linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Regressão Linear com Gradiente Descendente')
    plt.show()