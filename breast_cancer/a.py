import numpy as np

# Regressao Linear
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calcular_custo(theta, x, y):
    m = len(y)
    h = sigmoid(np.dot(x, theta))
    custo = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return custo

# Funcao de treinamento da regressão logística com Gradient Descent
def treinar_regressao_logistica(x, y, learning_rate, num_epochs):
    m, n = x.shape
    theta = np.zeros(n)

    for epoch in range(num_epochs):
        z = np.dot(x, theta)
        h = sigmoid(z)
        gradient = (1/m) * np.dot(x.T, (h - y))
        theta -= learning_rate * gradient

        custo = calcular_custo(theta, x, y)
        print(f'Época {epoch + 1}/{num_epochs}, Custo: {custo}')

    return theta
#================================================================================
# Discriminante Gaussiano
# Funcao para calcular os parametros do GDA
def calcular_parametros(x, y):
    classes = np.unique(y)
    parametros = []

    for classe in classes:
        x_classe = x[y == classe]
        media = np.mean(x_classe, axis=0)
        covariancia = np.cov(x_classe, rowvar=False)
        parametros.append((media, covariancia))

    return parametros

# Funcao para calcular a probabilidade da classe usando a distribuicao gaussiana
def calcular_probabilidade_classe(x, media, covariancia):
    n = len(media)
    det_cov = np.linalg.det(covariancia)
    inv_cov = np.linalg.pinv(covariancia)
    x_minus_media = x - media

    exponent = -0.5 * np.dot(x_minus_media, np.dot(inv_cov, x_minus_media))
    probabilidade = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_cov)) * np.exp(exponent)

    return probabilidade

# Funcao para fazer previsoes
def prever(x, parametros):
    previsoes = []
    for x in x:
        probabilidades_classe = []
        for media, covariancia in parametros:
            probabilidade = calcular_probabilidade_classe(x, media, covariancia)
            probabilidades_classe.append(probabilidade)
        classe_prevista = np.argmax(probabilidades_classe)
        previsoes.append(classe_prevista)
    return previsoes
#================================================================================

#Naive Bayes Gaussiano
# Funcao para calcular os parametros do Naive Bayes Gaussiano
def calcular_parametros(x, y):
    classes = np.unique(y)
    parametros = []

    for classe in classes:
        x_classe = x[y == classe]
        media = np.mean(x_classe, axis=0)
        desvio_padrao = np.std(x_classe, axis=0)
        parametros.append((media, desvio_padrao))

    return parametros

# Funcao para calcular a probabilidade de uma amostra pertencer a uma classe
def calcular_probabilidade_classe(x, media, desvio_padrao):
    probabilidade = 1.0
    for i in range(len(x)):
        probabilidade *= (1 / (np.sqrt(2 * np.pi) * desvio_padrao[i])) * \
                        np.exp(-((x[i] - media[i]) ** 2) / (2 * (desvio_padrao[i] ** 2)))
    return probabilidade

# Funcao para fazer previsoes
def prever(x, parametros):
    previsoes = []
    for x in x:
        probabilidades_classe = []
        for media, desvio_padrao in parametros:
            probabilidade = calcular_probabilidade_classe(x, media, desvio_padrao)
            probabilidades_classe.append(probabilidade)
        classe_prevista = np.argmax(probabilidades_classe)
        previsoes.append(classe_prevista)
    return previsoes
#================================================================================

#KNN
def distancia_euclidiana(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Funcao para encontrar os k vizinhos mais proximos
def encontrar_vizinhos(x_treino, ponto, k):
    distancias = []
    for i, x in enumerate(x_treino):
        dist = distancia_euclidiana(ponto, x)
        distancias.append((i, dist))
    distancias.sort(key=lambda x: x[1])
    vizinhos = [x[0] for x in distancias[:k]]
    return vizinhos

# Funcao para fazer a classificacao com base nos vizinhos
def classificar(x_treino, y_treino, ponto, k):
    vizinhos = encontrar_vizinhos(x_treino, ponto, k)
    classes_vizinhos = [y_treino[i] for i in vizinhos]
    classe_mais_comum = np.bincount(classes_vizinhos).argmax()
    return classe_mais_comum