import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """
    A K-Nearest Neighbors Classifier implemented from scratch using NumPy.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for queries.
    metric : str, default='euclidean'
        Distance metric to use. Supported metrics: 'euclidean', 'manhattan'.
    """
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        if self.metric not in ('euclidean', 'manhattan'):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        if self.X_train.ndim != 2:
            raise ValueError("X must be a 2D array-like structure")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same number of samples")
        if self.n_neighbors > len(self.X_train):
            raise ValueError("n_neighbors cannot be greater than the number of training samples")

        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, row):
        """Compute distances from a query row to all training samples."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - row) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - row), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _nearest_neighbor_labels(self, row):
        """Get the labels of the n_neighbors nearest neighbors for a single sample."""
        distances = self._compute_distances(row)
        # Argsort to find the indices of the sorted distances
        neighbor_indices = np.argsort(distances)[:self.n_neighbors]
        return self.y_train[neighbor_indices]

    def _predict_one(self, row):
        """Predict class for a single query sample."""
        neighbor_labels = self._nearest_neighbor_labels(row)
        values, counts = np.unique(neighbor_labels, return_counts=True)
        
        # Tie-breaking: choose the class with the maximum vote count.
        # If there is a tie, choose the first one (smallest label value).
        max_count = counts.max()
        tied_classes = values[counts == max_count]
        return tied_classes[0]

    def _predict_proba_one(self, row):
        """Predict class probabilities for a single query sample."""
        neighbor_labels = self._nearest_neighbor_labels(row)
        # Count occurrences of each class among the neighbors
        probs = []
        for c in self.classes_:
            probs.append(np.mean(neighbor_labels == c))
        return np.array(probs)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row) for row in X_arr])

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities for each data sample.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_one(row) for row in X_arr])

    def _check_is_fitted(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("This KNNClassifier instance is not fitted yet. Call 'fit' before predicting.")
