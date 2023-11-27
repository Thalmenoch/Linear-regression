import numpy as np

class Kmeans:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def kmeans_euclidiano(self, k, max_iters=100):

        centroids = self.dataframe.sample(k, random_state=42)

        for _ in range(max_iters):
        
            distances = np.linalg.norm(self.dataframe.values[:, np.newaxis] - centroids.values, axis=2)

            labels = np.argmin(distances, axis=1)

            centroids = self.dataframe.groupby(labels).mean()

        erro_de_reconstrucao = np.sum((self.dataframe.values - centroids.values[labels][:, np.newaxis]) ** 2)

        return labels, centroids, erro_de_reconstrucao

    def davies_bouldin_indice(self, labels, centroids):
        k = len(centroids)
        distances = np.linalg.norm(self.dataframe.values[:, np.newaxis] - centroids.values, axis=2)

        distancia_intra_cluster = np.zeros(k)
        for i in range(k):
            pontos_de_cluster = self.dataframe[labels == i]
            distancia_intra_cluster[i] = np.mean(np.linalg.norm(pontos_de_cluster.values - centroids.values[i], axis=1))

        distancia_intra_cluster = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                distancia_intra_cluster[i, j] = np.linalg.norm(centroids.values[i] - centroids.values[j])

        db_valores = np.zeros(k)
        for i in range(k):
            db_valores[i] = np.sum((distancia_intra_cluster[i] + distancia_intra_cluster) / distancia_intra_cluster[i, :])

        return np.mean(np.max(db_valores, axis=0))