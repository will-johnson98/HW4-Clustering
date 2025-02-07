import numpy as np
import pytest
from cluster.utils import make_clusters
from cluster.kmeans import KMeans

def test_kmeans_init():
    """Test KMeans initialization and input validation."""
    # Valid initialization
    kmeans = KMeans(k=3)
    assert kmeans.k == 3
    
    # Invalid k
    with pytest.raises(ValueError):
        KMeans(k=0)
    with pytest.raises(ValueError):
        KMeans(k=-1)
    with pytest.raises(TypeError):
        KMeans(k=3.5)
        
    # Invalid tolerance
    with pytest.raises(ValueError):
        KMeans(k=3, tol=-1e-6)
        
    # Invalid max_iter
    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=0)

def test_kmeans_fit_predict():
    """Test KMeans fit and predict methods."""
    # Create synthetic clusters
    X, true_labels = make_clusters(n=100, m=2, k=3, scale=0.3, seed=42)
    
    # Test basic fit/predict
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    pred_labels = kmeans.predict(X)
    
    assert pred_labels.shape == true_labels.shape
    assert len(np.unique(pred_labels)) == 3
    
    # Test convergence
    error = kmeans.get_error()
    assert error >= 0
    
    # Test centroids
    centroids = kmeans.get_centroids()
    assert centroids.shape == (3, 2)
    
def test_kmeans_edge_cases():
    """Test KMeans edge cases."""
    kmeans = KMeans(k=2)
    
    # Invalid fit input
    with pytest.raises(ValueError):
        kmeans.fit(np.array([1, 2, 3]))  # 1D array
        
    # Too few samples
    with pytest.raises(ValueError):
        kmeans.fit(np.random.randn(1, 2))
        
    # Predict before fit
    with pytest.raises(RuntimeError):
        kmeans.predict(np.random.randn(10, 2))
        
    # Wrong feature dimension in predict
    kmeans.fit(np.random.randn(10, 2))
    with pytest.raises(ValueError):
        kmeans.predict(np.random.randn(10, 3))
        
def test_kmeans_performance():
    """Test KMeans on different dataset sizes and dimensions."""
    # Test with 1D data
    X_1d, _ = make_clusters(n=100, m=1, k=2)
    kmeans = KMeans(k=2)
    kmeans.fit(X_1d)
    pred_1d = kmeans.predict(X_1d)
    assert pred_1d.shape == (100,)
    
    # Test with high-dimensional data
    X_hd, _ = make_clusters(n=1000, m=50, k=3)
    kmeans = KMeans(k=3)
    kmeans.fit(X_hd)
    pred_hd = kmeans.predict(X_hd)
    assert pred_hd.shape == (1000,)
    
    # Test with large k
    X, _ = make_clusters(n=1000, m=2, k=10)
    kmeans = KMeans(k=10)
    kmeans.fit(X)
    pred = kmeans.predict(X)
    assert len(np.unique(pred)) == 10