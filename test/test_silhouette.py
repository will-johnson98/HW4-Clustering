import numpy as np
import pytest
from sklearn.metrics import silhouette_score as sk_silhouette_score
from cluster.utils import make_clusters
from cluster.silhouette import Silhouette
from cluster.kmeans import KMeans

def test_silhouette_init():
    """Test Silhouette initialization."""
    silhouette = Silhouette()
    assert isinstance(silhouette, Silhouette)

def test_silhouette_score_basic():
    """Test basic Silhouette scoring functionality."""
    # Create synthetic clusters
    X, labels = make_clusters(n=100, m=2, k=3, scale=0.3, seed=42)
    silhouette = Silhouette()
    
    # Calculate scores
    scores = silhouette.score(X, labels)
    
    # Check shape and range
    assert scores.shape == (100,)
    assert np.all(scores >= -1)
    assert np.all(scores <= 1)

def test_silhouette_against_sklearn():
    """Test Silhouette score against sklearn implementation."""
    # Create synthetic clusters with different parameters
    test_cases = [
        (100, 2, 3, 0.3),  # Tight clusters
        (100, 2, 3, 2.0),  # Loose clusters
        (200, 5, 4, 1.0),  # Higher dimensions
        (500, 2, 10, 1.0), # More clusters
    ]
    
    silhouette = Silhouette()
    
    for n, m, k, scale in test_cases:
        X, labels = make_clusters(n=n, m=m, k=k, scale=scale, seed=42)
        
        # Calculate scores using both implementations
        our_scores = silhouette.score(X, labels)
        sk_scores = sk_silhouette_score(X, labels, metric='euclidean')
        
        # Compare individual scores
        assert np.allclose(np.mean(our_scores), sk_scores, rtol=1e-2)

def test_silhouette_edge_cases():
    """Test Silhouette score edge cases."""
    silhouette = Silhouette()
    X = np.random.randn(10, 2)
    
    # Invalid input dimensions
    with pytest.raises(ValueError):
        silhouette.score(np.array([1, 2, 3]), np.array([0, 0, 0]))  # 1D X
        
    # Mismatched dimensions
    with pytest.raises(ValueError):
        silhouette.score(X, np.array([0, 0]))  # Wrong label length
        
    # Single cluster
    with pytest.raises(ValueError):
        silhouette.score(X, np.zeros(10))  # All points in one cluster
        
def test_silhouette_with_kmeans():
    """Test Silhouette score with KMeans clustering results."""
    # Create data and cluster it
    X, _ = make_clusters(n=100, m=2, k=3, scale=0.3, seed=42)
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    pred_labels = kmeans.predict(X)
    
    # Calculate silhouette scores
    silhouette = Silhouette()
    scores = silhouette.score(X, pred_labels)
    
    # Verify scores
    assert scores.shape == (100,)
    assert np.all(scores >= -1)
    assert np.all(scores <= 1)
    
    # Compare with sklearn
    sk_scores = sk_silhouette_score(X, pred_labels, metric='euclidean')
    assert np.allclose(np.mean(scores), sk_scores, rtol=1e-2)