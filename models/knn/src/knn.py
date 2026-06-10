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
        
        # Vectorized distance calculation
        # Shape: (M, N) where M = len(X_array), N = len(self.X_train)
        distances = np.sqrt(np.sum((X_array[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        neighbor_labels = self.y_train[neighbor_indices]  # Shape: (M, k)
        
        # Majority vote for each prediction
        predictions = []
        for row in neighbor_labels:
            values, counts = np.unique(row, return_counts=True)
            max_count = counts.max()
            tied_values = values[counts == max_count]
            predictions.append(tied_values.min())
        return np.array(predictions)

    def predict_proba(self, X):
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        
        # Vectorized distance calculation
        distances = np.sqrt(np.sum((X_array[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        neighbor_labels = self.y_train[neighbor_indices]  # Shape: (M, k)
        
        probs = []
        for class_value in self.classes_:
            probs.append(np.mean(neighbor_labels == class_value, axis=1))
        return np.column_stack(probs)

    def _check_is_fitted(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("KNNClassifier must be fitted before prediction")
