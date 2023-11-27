import numpy as np

class Kmeans:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def kmeans_euclidiano(self, k, max_iters=100):
        # Inicializar os centróides aleatoriamente
        centroids = self.dataframe.sample(k, random_state=42)

        for _ in range(max_iters):
            # Calcular distâncias euclidianas entre pontos e centróides
            distances = np.linalg.norm(self.dataframe.values[:, np.newaxis] - centroids.values, axis=2)

            # Atribuir rótulos com base nas menores distâncias
            labels = np.argmin(distances, axis=1)

            # Atualizar os centróides com a média dos pontos em cada cluster
            centroids = self.dataframe.groupby(labels).mean()

        # Calcular o erro de reconstrução (soma dos quadrados das distâncias)
        erro_de_reconstrucao = np.sum((self.dataframe.values - centroids.values[labels][:, np.newaxis]) ** 2)

        return labels, centroids, erro_de_reconstrucao

    def davies_bouldin_indice(self, labels, centroids):
        k = len(centroids)
        distances = np.linalg.norm(self.dataframe.values[:, np.newaxis] - centroids.values, axis=2)

        # Calcular a dispersão intra-cluster (intra-cluster distance)
        distancia_intra_cluster = np.zeros(k)
        for i in range(k):
            pontos_de_cluster = self.dataframe[labels == i]
            distancia_intra_cluster[i] = np.mean(np.linalg.norm(pontos_de_cluster.values - centroids.values[i], axis=1))

        # Calcular a dispersão média inter-cluster (average inter-cluster distance)
        distancia_intra_cluster = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                distancia_intra_cluster[i, j] = np.linalg.norm(centroids.values[i] - centroids.values[j])

        # Calcular o índice Davies-Bouldin
        db_valores = np.zeros(k)
        for i in range(k):
            db_valores[i] = np.sum((distancia_intra_cluster[i] + distancia_intra_cluster) / distancia_intra_cluster[i, :])

        return np.mean(np.max(db_valores, axis=0))