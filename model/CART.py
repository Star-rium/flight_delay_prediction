import numpy as np


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        *,
        value=None,
        n_samples=0,
        gini=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.gini = gini

    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, class_weight=None):
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer greater than or equal to 2")
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.root = None
        self.feature_names = None
        self.classes_ = None
        self.class_weights_ = None

    @staticmethod
    def gini(y, sample_weight=None):
        if len(y) == 0:
            raise ValueError("Gini impurity is undefined for an empty label array")
        _, inverse = np.unique(y, return_inverse=True)
        if sample_weight is None:
            counts = np.bincount(inverse)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim != 1 or len(sample_weight) != len(y):
                raise ValueError("sample_weight must contain one weight per label")
            if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0):
                raise ValueError("sample_weight must contain finite non-negative values")
            counts = np.bincount(inverse, weights=sample_weight)
            if counts.sum() == 0:
                raise ValueError("sample_weight must have a positive sum")
        probabilities = counts / counts.sum()
        return float(1.0 - np.sum(probabilities**2))

    @staticmethod
    def split(col, threshold):
        return np.flatnonzero(col <= threshold), np.flatnonzero(col > threshold)

    def best_split(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        classes, encoded_y = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim != 1 or len(sample_weight) != n_samples:
                raise ValueError("sample_weight must contain one weight per sample")
            if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0):
                raise ValueError("sample_weight must contain finite non-negative values")
            if sample_weight.sum() == 0:
                raise ValueError("sample_weight must have a positive sum")

        parent_gini = self.gini(y, sample_weight)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            order = np.argsort(X[:, feature], kind="stable")
            sorted_values = X[order, feature]
            boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
            if boundaries.size == 0:
                continue

            sorted_labels = encoded_y[order]
            sorted_weights = sample_weight[order]
            one_hot = np.eye(n_classes, dtype=float)[sorted_labels] * sorted_weights[:, None]
            cumulative_counts = np.cumsum(one_hot, axis=0)
            left_counts = cumulative_counts[boundaries]
            right_counts = cumulative_counts[-1] - left_counts
            left_sizes = np.sum(left_counts, axis=1)
            right_sizes = np.sum(right_counts, axis=1)

            left_impurity = 1.0 - np.sum(left_counts**2, axis=1) / left_sizes**2
            right_impurity = 1.0 - np.sum(right_counts**2, axis=1) / right_sizes**2
            gains = parent_gini - (
                left_sizes * left_impurity + right_sizes * right_impurity
            ) / np.sum(sample_weight)

            candidate = int(np.argmax(gains))
            if gains[candidate] > best_gain:
                best_gain = float(gains[candidate])
                best_feature = feature
                best_threshold = sorted_values[boundaries[candidate]]

        return best_feature, best_threshold

    @staticmethod
    def _most_common(y, sample_weight=None):
        classes, inverse = np.unique(y, return_inverse=True)
        counts = np.bincount(inverse, weights=sample_weight)
        return classes[np.argmax(counts)]

    def _grow_tree(self, X, y, sample_weight, depth):
        n_samples = len(y)
        node_gini = self.gini(y, sample_weight)

        if (
            depth >= self.max_depth
            or node_gini == 0.0
            or n_samples < self.min_samples_split
        ):
            return Node(
                value=self._most_common(y, sample_weight),
                n_samples=n_samples,
                gini=node_gini,
            )

        feature, threshold = self.best_split(X, y, sample_weight)
        if feature is None:
            return Node(
                value=self._most_common(y, sample_weight),
                n_samples=n_samples,
                gini=node_gini,
            )

        left_mask = X[:, feature] <= threshold
        left = self._grow_tree(
            X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1
        )
        right = self._grow_tree(
            X[~left_mask], y[~left_mask], sample_weight[~left_mask], depth + 1
        )
        return Node(
            feature,
            threshold,
            left,
            right,
            n_samples=n_samples,
            gini=node_gini,
        )

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numeric values")
        if y.size == 0 or np.unique(y).size == 0:
            raise ValueError("y must contain at least one class label")
        if feature_names is not None and len(feature_names) != X.shape[1]:
            raise ValueError("feature_names must contain one name per feature")

        self.classes_, class_counts = np.unique(y, return_counts=True)
        if self.class_weight == "balanced":
            self.class_weights_ = len(y) / (len(self.classes_) * class_counts)
            class_weight_by_label = dict(zip(self.classes_, self.class_weights_))
            sample_weight = np.asarray(
                [class_weight_by_label[label] for label in y], dtype=float
            )
        else:
            self.class_weights_ = np.ones(len(self.classes_), dtype=float)
            sample_weight = np.ones(len(y), dtype=float)

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(X.shape[1])]
        )
        self.root = self._grow_tree(X, y, sample_weight, depth=0)
        return self

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        if self.root is None or self.feature_names is None:
            raise ValueError("The decision tree must be fitted before calling predict")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[1] != len(self.feature_names):
            raise ValueError("X must contain the same features used during fitting")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numeric values")
        return np.asarray([self._traverse(row, self.root) for row in X])

    def get_depth(self):
        """Return the maximum number of edges from the root to a leaf."""
        if self.root is None:
            raise ValueError("The decision tree must be fitted before getting its depth")

        def depth(node):
            if node.is_leaf():
                return 0
            return 1 + max(depth(node.left), depth(node.right))

        return depth(self.root)

    def get_n_leaves(self):
        """Return the number of terminal nodes in the fitted tree."""
        if self.root is None:
            raise ValueError("The decision tree must be fitted before counting leaves")

        def count(node):
            if node.is_leaf():
                return 1
            return count(node.left) + count(node.right)

        return count(self.root)
