import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calcular_custo(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    custo = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return custo

# Função de treinamento da regressão logística com Gradient Descent
def treinar_regressao_logistica(X, y, learning_rate, num_epochs):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(num_epochs):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient

        custo = calcular_custo(theta, X, y)
        print(f'Época {epoch + 1}/{num_epochs}, Custo: {custo}')

    return theta