"""
K-Means image segmentation module in CIELAB color space.
"""
from typing import Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def segment_kmeans(
    lab_image: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment an image using K-Means clustering on CIELAB color features.
    
    Args:
        lab_image: (H, W, 3) CIELAB format image array.
        n_clusters: Number of clusters (typically 3, 5, or 7).
        random_state: Random seed for reproducibility.
        batch_size: Batch size for MiniBatchKMeans for performance.
        
    Returns:
        Tuple of:
            - cluster_labels: (H, W) array with cluster indices [0, n_clusters - 1].
            - cluster_centers: (n_clusters, 3) cluster centroids in CIELAB space.
    """
    if not isinstance(lab_image, np.ndarray) or lab_image.ndim != 3 or lab_image.shape[2] != 3:
        raise ValueError("lab_image must be a 3D numpy array with 3 channels.")

    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

    h, w, c = lab_image.shape
    features = lab_image.reshape(-1, c).astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=3,
        max_iter=100,
    )
    labels_1d = kmeans.fit_predict(features)
    cluster_labels = labels_1d.reshape((h, w)).astype(np.int32)
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers
