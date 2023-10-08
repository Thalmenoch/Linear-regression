import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Função de cálculo do RMSE (Root Mean Squared Error)
def RMSE(y, y_previsto):
    return np.sqrt(np.mean((y - y_previsto) ** 2))

# Função para calcular os coeficientes com regularização Ridge (LS regularizado)
def regularizacao(x_treino_poly, y_treino, alpha):
    I = np.identity(x_treino_poly.shape[1])
    coeficientes = np.linalg.inv(x_treino_poly.T @ x_treino_poly + alpha * I) @ x_treino_poly.T @ y_treino
    return coeficientes 

# Função para treinar o modelo
def treinamento_do_modelo(x_treino_norm, x_teste_norm, y_treino, y_teste, alpha):
    lista_rmse_treino = []
    lista_rmse_teste = []

    for ordem in range(1, 12):
        X_treino_poly = np.column_stack([x_treino_norm ** i for i in range(1, ordem + 1)])
        X_teste_poly = np.column_stack([x_teste_norm ** i for i in range(1, ordem + 1)])
        
        coeficientes = regularizacao(X_treino_poly, y_treino, alpha)
        
        y_pred_treino = X_treino_poly @ coeficientes
        y_pred_teste = X_teste_poly @ coeficientes
        
        y_pred_desnormalizado_treino = scaler_y.inverse_transform(y_pred_treino.reshape(-1, 1)).flatten()
        y_pred_desnormalizado_teste = scaler_y.inverse_transform(y_pred_teste.reshape(-1, 1)).flatten()
        
        y_treino_desnormalizado = scaler_y.inverse_transform(y_treino)
        y_teste_desnormalizado = scaler_y.inverse_transform(y_teste)
        
        rmse_treino = RMSE(y_treino_desnormalizado, y_pred_desnormalizado_treino)
        rmse_teste = RMSE(y_teste_desnormalizado, y_pred_desnormalizado_teste)
        
        lista_rmse_treino.append(rmse_treino)
        lista_rmse_teste.append(rmse_teste)

        print(f"Ordem do Polinômio: {ordem}")
        print(f"RMSE no treino: {rmse_treino:.2f}")
        print(f"RMSE no teste: {rmse_teste:.2f}")
        print("=" * 40)

    return lista_rmse_treino, lista_rmse_teste

if __name__ == '__main__':
    data = pd.read_csv('regressao_polinomial/boston.csv')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler_x = MinMaxScaler()
    x_treino_norm = scaler_x.fit_transform(x_treino)
    x_teste_norm = scaler_x.transform(x_teste)

    scaler_y = MinMaxScaler()
    y_treino_norm = scaler_y.fit_transform(y_treino.values.reshape(-1, 1))
    y_teste_norm = scaler_y.transform(y_teste.values.reshape(-1, 1))

    alpha = 1.0  # Parâmetro de regularização (ajuste conforme necessário)

    lista_rmse_treino, lista_rmse_teste = treinamento_do_modelo(x_treino_norm, x_teste_norm, y_treino_norm, y_teste_norm, alpha)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 12), lista_rmse_treino, marker='o', label='Treinamento')
    plt.title('RMSE no Treinamento')
    plt.xlabel('Ordem do Polinômio')
    plt.ylabel('RMSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 12), lista_rmse_teste, marker='o', label='Teste')
    plt.title('RMSE no Teste')
    plt.xlabel('Ordem do Polinômio')
    plt.ylabel('RMSE')
    plt.legend()

    plt.tight_layout()
    plt.show()
