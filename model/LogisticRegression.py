import numpy as np

class LogisticRegressionScratch:
    def __init__(self, eta, max_iter, tol = 1e-4, class_weight=None, random_state=42):
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500))) # clip the value so the number doesnt get too high

    def add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def get_sample_weights(self, y):
        y = np.asarray(y)
        if self.class_weight == "balanced":
            n = len(y)
            n_pos = y.sum()
            n_neg = n - n_pos
            return np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))
        return np.ones(len(y))

    def compute_loss(self, X, y, w, sw):
        z = self.sigmoid(np.dot(X, w))
        loss = -np.mean(sw * (y * np.log(z + 1e-15) + (1 - y) * np.log(1 - z + 1e-15))) # add small term to avoid log(0)
        return loss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = self.add_bias(X)
        N, d = Xb.shape
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d)                # loss is convex, so zeros is a fine start
        weighted_label = self.get_sample_weights(y)

        self.loss_history = [self.compute_loss(Xb, y, self.w, weighted_label)]

        for _ in range(self.max_iter):
            for i in rng.permutation(N):    # shuffle every epoch
                xi = Xb[i]
                zi = self.sigmoid(np.dot(xi, self.w))
                self.w += self.eta * weighted_label[i] * (y[i] - zi) * xi # gradient update
 
            loss = self.compute_loss(Xb, y, self.w, weighted_label)
            self.loss_history.append(loss)
            if np.linalg.norm(self.loss_history[-2] - loss) < self.tol:
                break
        return self

    def predict_proba(self, X):
        Xb = self.add_bias(np.asarray(X, dtype=float))
        return self.sigmoid(np.dot(Xb, self.w))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
