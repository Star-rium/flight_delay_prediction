import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestScratch:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt", class_weight=None, oob_score=False,
                    random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features        # "sqrt" -> m = sqrt(p) features per split
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.random_state = random_state
        self.trees = []
        self.oob_proba_ = None                  # OOB probability for every training row
        self.feature_importances_ = None

    def get_sample_weights(self, y):
        # return weighted class to avoid class imbalance
        n = len(y)
        n_pos = y.sum()
        n_neg = n - n_pos
        return np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))

    def _tree_proba(self, tree, X):
        # return probability of class 1 in the tree
        p = tree.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return p[:, 0] if tree.classes_[0] == 1 else np.zeros(len(X))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        sw = self.get_sample_weights(y) if self.class_weight == "balanced" else None

        self.trees = []
        importance = np.zeros(p) # initialize feature importance vector
        if self.oob_score:
            oob_sum = np.zeros(n)
            oob_cnt = np.zeros(n)

        for _ in range(self.n_estimators):
            # bootstrap aggregating
            idx = rng.integers(0, n, size=n)
            tree = DecisionTreeClassifier(
                criterion="gini",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
            tree.fit(X[idx], y[idx], sample_weight=None if sw is None else sw[idx])
            self.trees.append(tree)
            importance += tree.feature_importances_

            if self.oob_score:
                in_bag = np.zeros(n, dtype=bool)
                in_bag[idx] = True
                oob_idx = np.flatnonzero(~in_bag) # rows this tree never saw (~36.8%)
                oob_sum[oob_idx] += self._tree_proba(tree, X[oob_idx])
                oob_cnt[oob_idx] += 1

        self.feature_importances_ = importance / self.n_estimators
        if self.oob_score:
            with np.errstate(invalid="ignore", divide="ignore"):
                self.oob_proba_ = oob_sum / oob_cnt  # NaN if a row was never out-of-bag
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        total = np.zeros(X.shape[0])
        for tree in self.trees:
            total += self._tree_proba(tree, X)
        return total / len(self.trees)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def staged_predict_proba(self, X):
        # yield the forest's probability after each trees
        X = np.asarray(X, dtype=float)
        total = np.zeros(X.shape[0])
        for t, tree in enumerate(self.trees, start=1):
            total += self._tree_proba(tree, X)
            yield total / t