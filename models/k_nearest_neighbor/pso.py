import numpy as np

class BinaryPSOFeatureSelection:
    """
    Binary Particle Swarm Optimization (PSO) for Feature Selection.
    
    Parameters:
    -----------
    num_features : int
        Total number of features in the dataset.
    estimator : estimator object
        An instance of a classifier supporting 'fit' and 'predict' methods.
    num_particles : int, default=15
        Number of particles in the swarm.
    max_iter : int, default=20
        Maximum number of iterations.
    w : float, default=0.9
        Inertia weight.
    c1 : float, default=2.0
        Cognitive coefficient.
    c2 : float, default=2.0
        Social coefficient.
    alpha : float, default=0.99
        Weight parameter for accuracy vs number of selected features.
        Fitness = alpha * Accuracy + (1 - alpha) * (1 - N_selected / N_total)
    verbose : bool, default=True
        Whether to print progress messages.
    """
    def __init__(self, num_features, estimator, num_particles=15, max_iter=20, w=0.9, c1=2.0, c2=2.0, alpha=0.99, verbose=True):
        self.num_features = num_features
        self.estimator = estimator
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.verbose = verbose
        
        # Initialize particles' positions: shape (num_particles, num_features)
        # Random binary choice: each feature has a 50% chance of being selected (1)
        self.X_pos = np.random.choice([0, 1], size=(self.num_particles, self.num_features))
        
        # Ensure at least one feature is selected for each particle
        for i in range(self.num_particles):
            if np.sum(self.X_pos[i]) == 0:
                self.X_pos[i, np.random.randint(self.num_features)] = 1
                
        # Initialize velocities randomly in range [-4.0, 4.0]
        self.V = np.random.uniform(-4.0, 4.0, size=(self.num_particles, self.num_features))
        
        # Personal best position and fitness
        self.pbest_pos = np.copy(self.X_pos)
        self.pbest_fitness = np.zeros(self.num_particles)
        
        # Global best position and fitness
        self.gbest_pos = np.zeros(self.num_features)
        self.gbest_fitness = -1.0

    def _sigmoid(self, x):
        # Clip values to prevent overflow in exp
        x = np.clip(x, -100, 100)
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Run the PSO feature selection.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training labels.
        X_val : array-like of shape (n_samples, n_features)
            Validation feature matrix.
        y_val : array-like of shape (n_samples,)
            Validation labels.
            
        Returns:
        --------
        best_features : ndarray of shape (n_features,)
            Binary mask of selected features (1 for selected, 0 for dropped).
        best_fitness : float
            Best fitness achieved.
        """
        for it in range(self.max_iter):
            for i in range(self.num_particles):
                # Get indices of selected features
                selected_features = np.where(self.X_pos[i] == 1)[0]
                
                if len(selected_features) == 0:
                    fitness = 0.0
                else:
                    # Subset features
                    X_tr_sub = X_train[:, selected_features]
                    X_va_sub = X_val[:, selected_features]
                    
                    # Fit and evaluate
                    self.estimator.fit(X_tr_sub, y_train)
                    y_pred = self.estimator.predict(X_va_sub)
                    acc = np.mean(y_pred == y_val)
                    
                    # Fitness function with penalty for selected feature counts
                    # Fitness = alpha * Accuracy + (1 - alpha) * (1 - N_selected / N_total)
                    fitness = self.alpha * acc + (1 - self.alpha) * (1 - len(selected_features) / self.num_features)
                
                # Update personal best
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_pos[i] = np.copy(self.X_pos[i])
                    
                # Update global best
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_pos = np.copy(self.X_pos[i])
            
            # Update velocity and position for the next iteration
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.num_features), np.random.rand(self.num_features)
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.X_pos[i])
                social = self.c2 * r2 * (self.gbest_pos - self.X_pos[i])
                self.V[i] = self.w * self.V[i] + cognitive + social
                
                # Clip velocity
                self.V[i] = np.clip(self.V[i], -6.0, 6.0)
                
                # Sigmoid transfer function for binary selection
                probs = self._sigmoid(self.V[i])
                rand_vals = np.random.rand(self.num_features)
                self.X_pos[i] = np.where(rand_vals < probs, 1, 0)
                
                # Force at least one feature to be selected
                if np.sum(self.X_pos[i]) == 0:
                    self.X_pos[i, np.random.randint(self.num_features)] = 1
            
            if self.verbose:
                selected_indices = np.where(self.gbest_pos == 1)[0]
                # Calculate actual validation accuracy (without penalty) for progress printing
                if len(selected_indices) > 0:
                    self.estimator.fit(X_train[:, selected_indices], y_train)
                    val_acc = np.mean(self.estimator.predict(X_val[:, selected_indices]) == y_val)
                else:
                    val_acc = 0.0
                print(f"Iteration {it+1:2d}/{self.max_iter}: Best Val Acc = {val_acc:.4f} | Selected features indices = {selected_indices.tolist()}")
                
        return self.gbest_pos, self.gbest_fitness
