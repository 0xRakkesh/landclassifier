"""
Image segmentation module: K-Means and DBSCAN density-based clustering in CIELAB color space.
"""
from typing import Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

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


def segment_dbscan(
    lab_image: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 25,
    sample_size: int = 6000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment an image using DBSCAN density-based clustering on CIELAB features.
    Uses representative sub-sampling + Nearest-Neighbors propagation for interactive performance.
    
    Args:
        lab_image: (H, W, 3) CIELAB format image array.
        eps: Maximum distance between two samples for them to be considered in the same neighborhood.
        min_samples: Minimum number of samples in a neighborhood for a point to be a core point.
        sample_size: Number of pixels sampled to fit DBSCAN in real time.
        random_state: Seed for sampling.
        
    Returns:
        Tuple of:
            - cluster_labels: (H, W) array with non-negative cluster IDs.
            - cluster_centers: (K, 3) cluster centroid positions in CIELAB space.
    """
    if not isinstance(lab_image, np.ndarray) or lab_image.ndim != 3 or lab_image.shape[2] != 3:
        raise ValueError("lab_image must be a 3D numpy array with 3 channels.")

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if min_samples < 2:
        raise ValueError(f"min_samples must be >= 2, got {min_samples}")

    h, w, c = lab_image.shape
    total_pixels = h * w
    features = lab_image.reshape(-1, c).astype(np.float32)

    # Subsample for responsive clustering
    if total_pixels > sample_size:
        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(total_pixels, size=sample_size, replace=False)
        sub_features = features[sample_indices]
    else:
        sub_features = features

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    sub_labels = dbscan.fit_predict(sub_features)

    # Filter distinct non-noise clusters
    valid_clusters = [lbl for lbl in np.unique(sub_labels) if lbl != -1]

    if not valid_clusters:
        # Fallback: if everything is noise or uniform, treat whole image as single cluster
        cluster_labels = np.zeros((h, w), dtype=np.int32)
        cluster_centers = np.mean(features, axis=0, keepdims=True)
        return cluster_labels, cluster_centers

    # Map raw cluster IDs to clean 0..K-1
    lbl_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(sorted(valid_clusters))}
    
    # Propagate labels to all image pixels using 1-NN trained on valid core/sample points
    core_mask = sub_labels != -1
    X_core = sub_features[core_mask]
    y_core = np.array([lbl_map[l] for l in sub_labels[core_mask]], dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_core)
    distances, nn_indices = nn.kneighbors(features)
    
    assigned_labels = y_core[nn_indices.ravel()]
    
    # Points with very large distance (> eps * 2.5) are marked as noise
    noise_idx = len(lbl_map) # Next ID for noise / uncertain
    is_outlier = distances.ravel() > (eps * 2.5)
    assigned_labels[is_outlier] = noise_idx

    cluster_labels = assigned_labels.reshape((h, w)).astype(np.int32)

    # Compute centroids for each cluster
    n_total_clusters = noise_idx + (1 if np.any(is_outlier) else 0)
    cluster_centers = np.zeros((n_total_clusters, 3), dtype=np.float32)

    for cid in range(n_total_clusters):
        pts = features[assigned_labels == cid]
        if len(pts) > 0:
            cluster_centers[cid] = np.mean(pts, axis=0)
        else:
            cluster_centers[cid] = np.array([128.0, 128.0, 128.0], dtype=np.float32) # neutral gray

    return cluster_labels, cluster_centers
