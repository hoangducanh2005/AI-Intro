import numpy as np


class KNNClassifier:
    def __init__(self, k=5):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X_array) != len(y_array):
            raise ValueError("X and y must contain the same number of samples")
        if self.k > len(X_array):
            raise ValueError("k cannot be greater than the number of training samples")

        self.X_train = X_array
        self.y_train = y_array
        self.classes_ = np.unique(y_array)
        return self

    def predict(self, X):
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row) for row in X_array])

    def predict_proba(self, X):
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_one(row) for row in X_array])

    def _predict_one(self, row):
        neighbor_labels = self._nearest_neighbor_labels(row)
        values, counts = np.unique(neighbor_labels, return_counts=True)
        max_count = counts.max()
        tied_values = values[counts == max_count]
        return tied_values.min()

    def _predict_proba_one(self, row):
        neighbor_labels = self._nearest_neighbor_labels(row)
        probabilities = []
        for class_value in self.classes_:
            probabilities.append(np.mean(neighbor_labels == class_value))
        return np.array(probabilities)

    def _nearest_neighbor_labels(self, row):
        distances = np.sqrt(np.sum((self.X_train - row) ** 2, axis=1))
        neighbor_indices = np.argsort(distances)[: self.k]
        return self.y_train[neighbor_indices]

    def _check_is_fitted(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("KNNClassifier must be fitted before prediction")
