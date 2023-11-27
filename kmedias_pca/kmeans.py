import numpy as np

class Kmeans:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def kmeans_euclidean(self, k, max_iters=100):
        # Inicializar os centróides aleatoriamente
        centroids = self.dataframe.sample(k, random_state=42)

        for _ in range(max_iters):
            # Calcular distâncias euclidianas entre pontos e centróides
            distances    = np.linalg.norm(self.dataframe.values[:, np.newaxis] - centroids.values, axis=2)

            # Atribuir rótulos com base nas menores distâncias
            labels = np.argmin(distances, axis=1)

            # Atualizar os centróides com a média dos pontos em cada cluster
            centroids = self.dataframe.groupby(labels).mean()

        # Calcular o erro de reconstrução (soma dos quadrados das distâncias)
        reconstruction_error = np.sum((self.dataframe.values - centroids.values[labels][:, np.newaxis]) ** 2)

        return labels, centroids, reconstruction_error

    def davies_bouldin_index(self, labels, centroids):
        k = len(centroids)
        distances = np.linalg.norm(self.dataframe.values[:, np.newaxis] - centroids.values, axis=2)

        # Calcular a dispersão intra-cluster (intra-cluster distance)
        intra_cluster_distances = np.zeros(k)
        for i in range(k):
            cluster_points = self.dataframe[labels == i]
            intra_cluster_distances[i] = np.mean(np.linalg.norm(cluster_points.values - centroids.values[i], axis=1))

        # Calcular a dispersão média inter-cluster (average inter-cluster distance)
        inter_cluster_distances = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                inter_cluster_distances[i, j] = np.linalg.norm(centroids.values[i] - centroids.values[j])

        # Calcular o índice Davies-Bouldin
        db_values = np.zeros(k)
        for i in range(k):
            db_values[i] = np.sum((intra_cluster_distances[i] + intra_cluster_distances) / inter_cluster_distances[i, :])

        return np.mean(np.max(db_values, axis=0))