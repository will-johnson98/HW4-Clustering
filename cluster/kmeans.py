import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """Initialize KMeans clustering algorithm."""
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be positive")
        if tol <= 0:
            raise ValueError("tolerance must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
            
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.n_features = None
        self.error = None
        self._fitted = False
        
    def fit(self, mat: np.ndarray):
        """Fit KMeans to data using Lloyd's Algorithm."""
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input must be 2D numpy array")
        if mat.shape[0] < self.k:
            raise ValueError("Number of samples must be >= k")
            
        n_samples, n_features = mat.shape
        self.n_features = n_features
        
        # Initialize centroids randomly from data points
        rng = np.random.default_rng()
        init_indices = rng.choice(n_samples, size=self.k, replace=False)
        self.centroids = mat[init_indices].copy()
        
        prev_error = float('inf')
        self.error = float('inf')
        
        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            distances = cdist(mat, self.centroids)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for j in range(self.k):
                if np.sum(labels == j) > 0:  # Avoid empty clusters
                    self.centroids[j] = np.mean(mat[labels == j], axis=0)
                    
            # Calculate error
            self.error = np.mean(np.min(distances, axis=1) ** 2)
            
            # Check convergence
            if abs(prev_error - self.error) < self.tol:
                break
                
            prev_error = self.error
            
        self._fitted = True
        
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input must be 2D numpy array")
        if mat.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {mat.shape[1]}")
            
        distances = cdist(mat, self.centroids)
        return np.argmin(distances, axis=1)
        
    def get_error(self) -> float:
        """Return the final squared-mean error."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before getting error")
        return self.error
        
    def get_centroids(self) -> np.ndarray:
        """Return the cluster centroids."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before getting centroids")
        return self.centroids.copy()