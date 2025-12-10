import numpy as np

class MiniBatchKMeansScratch:
    def __init__(self, n_clusters=6, batch_size=5000, max_iter=20, random_state=42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)

    def initialize_centroids(self, X):
        idx = self.random_state.choice(len(X), self.n_clusters, replace=False)
        return X[idx].copy()

    def partial_fit(self, batch):
        # compute distances to centroids
        distances = np.sqrt(((batch[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        # online centroid update
        for k in range(self.n_clusters):
            points = batch[labels == k]
            if len(points) > 0:
                lr = 1.0 / (1 + self.counts[k])   # learning rate per cluster
                self.centroids[k] = (1 - lr) * self.centroids[k] + lr * points.mean(axis=0)
                self.counts[k] += 1

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        self.counts = np.zeros(self.n_clusters, dtype=np.int32)

        n = len(X)

        for it in range(self.max_iter):
            print(f"Iteration {it+1}/{self.max_iter}")

            # shuffle indexes
            idx = np.arange(n)
            self.random_state.shuffle(idx)

            # iterate over minibatches
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch = X[idx[start:end]]

                if len(batch) > 0:
                    self.partial_fit(batch)

        return self

    def predict(self, X):
        distances = np.sqrt(((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

