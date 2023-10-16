import numpy as np

# Função para calcular os parâmetros do GDA
def calcular_parametros(X, y):
    classes = np.unique(y)
    parametros = []

    for classe in classes:
        X_classe = X[y == classe]
        media = np.mean(X_classe, axis=0)
        covariancia = np.cov(X_classe, rowvar=False)
        parametros.append((media, covariancia))

    return parametros

# Função para calcular a probabilidade da classe usando a distribuição gaussiana
def calcular_probabilidade_classe(x, media, covariancia):
    n = len(media)
    det_cov = np.linalg.det(covariancia)
    inv_cov = np.linalg.pinv(covariancia)
    x_minus_media = x - media

    exponent = -0.5 * np.dot(x_minus_media, np.dot(inv_cov, x_minus_media))
    probabilidade = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_cov)) * np.exp(exponent)

    return probabilidade

# Função para fazer previsões
def prever(X, parametros):
    previsoes = []
    for x in X:
        probabilidades_classe = []
        for media, covariancia in parametros:
            probabilidade = calcular_probabilidade_classe(x, media, covariancia)
            probabilidades_classe.append(probabilidade)
        classe_prevista = np.argmax(probabilidades_classe)
        previsoes.append(classe_prevista)
    return previsoes