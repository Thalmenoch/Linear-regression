import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Cálculo do MSE
def MSE(y, y_previsto):
    return np.mean((y - y_previsto) ** 2)

#Cálculo do Gradiente Descendente Estocástico
def SGD(x, y, taxa_aprendizado, nmr_iteracoes):
    #Inicialização dos parâmetros do modelo
    coef_angular = 0
    vies = 0
    norm = len(x) #Normaliza o gradiente pela quantidade de pontos de dados
    
    #Lista para armazenar os valores do MSE
    lista_mse = []

    for _ in range(nmr_iteracoes):
        #Escolha aleatória de um ponto de dados 
        index = np.random.randint(0, norm)

        #Cálculo da previsão do modelo para o ponto de dados selecionado
        y_previsto = coef_angular * x[index] + vies

        #Cálculo do erro em relação ao ponto de dados
        erro = y_previsto - y[index]

        #Atualizando os parâmetros usando o gradiente
        gradiente_coef = 2 * erro * x[index]
        gradiente_vies = 2 * erro

        coef_angular -= taxa_aprendizado * gradiente_coef
        vies -= taxa_aprendizado * gradiente_vies

        #Cálculo do MSE atual e o armazenenamento dele na lista
        mse_atual = np.mean((y - (coef_angular * x + vies))**2)
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

    #Hiperparâmetros do Gradiente Descendente Estocástico
    taxa_aprendizado = 0.02
    # nmr_iteracoes = 100
    nmr_iteracoes = 500

    #Variáveis que recebem o Gradiente Descendente
    coef_angular, vies, lista_mse = SGD(x, y, taxa_aprendizado, nmr_iteracoes)

    #Print dos parâmetros do modelo 
    print(f"Coeficiente Angular: {coef_angular:.2f}")
    print(f"Viés: {vies:.2f}")
    print(f"MSE: {lista_mse[-1]:.2f}")

    #Plotando a curva de aprendizagem 
    plt.plot(range(1, nmr_iteracoes + 1), lista_mse)
    plt.xlabel('Número de Iterações')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.title('Curva de Aprendizagem do Gradiente Descendente Estocástico')
    plt.show()

    #Plotando a reta de regressão sobre os dados
    y_previsto = coef_angular * x + vies
    plt.scatter(x, y, label='Dados Recebidos')
    plt.plot(x, y_previsto, color='red', label='Regressão Linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Regressão Linear com Gradiente Descendente Estocástico')
    plt.show()