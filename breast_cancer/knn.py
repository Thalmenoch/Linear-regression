import numpy as np

def distancia_euclidiana(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Função para encontrar os k vizinhos mais próximos
def encontrar_vizinhos(X_treino, ponto, k):
    distancias = []
    for i, x in enumerate(X_treino):
        dist = distancia_euclidiana(ponto, x)
        distancias.append((i, dist))
    distancias.sort(key=lambda x: x[1])
    vizinhos = [x[0] for x in distancias[:k]]
    return vizinhos

# Função para fazer a classificação com base nos vizinhos
def classificar(X_treino, y_treino, ponto, k):
    vizinhos = encontrar_vizinhos(X_treino, ponto, k)
    classes_vizinhos = [y_treino[i] for i in vizinhos]
    classe_mais_comum = np.bincount(classes_vizinhos).argmax()
    return classe_mais_comum