import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X_array) != len(y_array):
            raise ValueError("X and y must contain the same number of samples")

        self.classes_ = np.unique(y_array)
        n_samples, n_features = X_array.shape
        n_classes = len(self.classes_)

        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))

        for idx, class_val in enumerate(self.classes_):
            X_class = X_array[y_array == class_val]
            
            # Tính xác suất tiên nghiệm P(Y)
            self.class_priors_[idx] = len(X_class) / n_samples
            
            # Tính trung bình và phương sai cho mỗi đặc trưng
            self.means_[idx] = np.mean(X_class, axis=0)
            
            # Cộng thêm một lượng nhỏ epsilon (1e-9) vào phương sai để tránh chia cho 0
            self.variances_[idx] = np.var(X_class, axis=0) + 1e-9

        return self

    def predict_joint_log_likelihood(self, X):
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        n_samples = X_array.shape[0]
        n_classes = len(self.classes_)
        
        joint_log_likelihood = np.zeros((n_samples, n_classes))

        for idx, _ in enumerate(self.classes_):
            mean = self.means_[idx]
            var = self.variances_[idx]
            prior = self.class_priors_[idx]

            # Công thức log của phân phối xác suất Gauss cho đặc trưng thứ i của mẫu x:
            # log P(x_i | Y = c) = -0.5 * log(2 * pi * var_i) - (x_i - mean_i)^2 / (2 * var_i)
            log_prior = np.log(prior)
            exponent = -((X_array - mean) ** 2) / (2 * var)
            log_likelihood_features = -0.5 * np.log(2 * np.pi * var) + exponent
            
            # Tổng log-likelihood các đặc trưng và cộng với log-prior
            joint_log_likelihood[:, idx] = log_prior + np.sum(log_likelihood_features, axis=1)

        return joint_log_likelihood

    def predict(self, X):
        joint_log_likelihood = self.predict_joint_log_likelihood(X)
        best_class_indices = np.argmax(joint_log_likelihood, axis=1)
        return self.classes_[best_class_indices]

    def predict_proba(self, X):
        joint_log_likelihood = self.predict_joint_log_likelihood(X)
        
        # Áp dụng mẹo Log-Sum-Exp để tránh tràn số:
        # Subtract max log-likelihood trước khi luỹ thừa cơ số e
        max_log = np.max(joint_log_likelihood, axis=1, keepdims=True)
        exp_shifted = np.exp(joint_log_likelihood - max_log)
        
        # Chuẩn hóa về [0, 1]
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return probs

    def _check_is_fitted(self):
        if self.classes_ is None or self.class_priors_ is None:
            raise ValueError("NaiveBayesClassifier must be fitted before prediction")
