import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self):
        """Initialize Silhouette scorer."""
        pass
        
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate silhouette score for each observation."""
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be 2D numpy array")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("y must be 1D numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")
        if len(np.unique(y)) < 2:
            raise ValueError("Number of clusters must be greater than 1")
            
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        # Compute pairwise distances between all points
        distances = cdist(X, X)
        
        for i in range(n_samples):
            # Get cluster label for current point
            current_cluster = y[i]
            
            # Find points in same cluster (excluding current point)
            same_cluster = (y == current_cluster) & (np.arange(n_samples) != i)
            
            if np.sum(same_cluster) > 0:
                # Calculate a(i): mean distance to points in same cluster
                a_i = np.mean(distances[i, same_cluster])
            else:
                # Handle singleton clusters
                a_i = 0
                
            # Calculate mean distances to each other cluster
            other_cluster_distances = []
            for cluster in np.unique(y):
                if cluster != current_cluster:
                    other_cluster = y == cluster
                    if np.sum(other_cluster) > 0:
                        mean_dist = np.mean(distances[i, other_cluster])
                        other_cluster_distances.append(mean_dist)
                        
            if other_cluster_distances:
                # Calculate b(i): mean distance to nearest neighboring cluster
                b_i = np.min(other_cluster_distances)
                
                # Calculate silhouette score
                max_ab = max(a_i, b_i)
                if max_ab > 0:
                    scores[i] = (b_i - a_i) / max_ab
                else:
                    scores[i] = 0
            else:
                # Handle case where there's only one cluster
                scores[i] = 0
                
        return scores