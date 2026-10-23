"""
High-accuracy K-Means image segmentation module.
Combines CIELAB color space, Normalized Excess Green (ExG), and Water Absorption Index (WAI)
with k-means++ centroid initialization for high classification accuracy.
"""
from typing import Tuple, Optional
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_spectral_features(
    lab_image: np.ndarray,
    bgr_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract multi-dimensional spectral feature representation per pixel:
    1. L* (Lightness)
    2. a* (Green-Red chromaticity)
    3. b* (Blue-Yellow chromaticity)
    4. ExG (Normalized Excess Green Index = 2G - R - B)
    5. WAI (Water Absorption Index = (B - R) / (B + R + eps))
    
    Returns:
        (H*W, 5) standardized feature matrix.
    """
    h, w, c = lab_image.shape
    total_pixels = h * w

    if bgr_image is None:
        bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    b = bgr_image[:, :, 0].astype(np.float32)
    g = bgr_image[:, :, 1].astype(np.float32)
    r = bgr_image[:, :, 2].astype(np.float32)

    L = lab_image[:, :, 0].astype(np.float32)
    a = lab_image[:, :, 1].astype(np.float32)
    lab_b = lab_image[:, :, 2].astype(np.float32)

    # 1. Excess Green Index: highlights vegetation
    exg = (2.0 * g - r - b) / 255.0

    # 2. Water Absorption Index: separates deep water from land
    wai = (b - r) / (b + r + 1e-4)

    # Combine into (H*W, 5) matrix
    features = np.column_stack([
        L.reshape(-1),
        a.reshape(-1),
        lab_b.reshape(-1),
        exg.reshape(-1),
        wai.reshape(-1),
    ])

    # Standardize features (zero mean, unit variance) so K-Means weights all spectral indices fairly
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features


def segment_kmeans(
    lab_image: np.ndarray,
    bgr_image: Optional[np.ndarray] = None,
    n_clusters: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment satellite image using high-precision K-Means with k-means++ seeding.
    
    Args:
        lab_image: (H, W, 3) CIELAB format image array.
        bgr_image: Optional (H, W, 3) BGR format image array.
        n_clusters: Number of clusters (typically 3, 5, or 7).
        random_state: Random seed for exact reproducibility.
        
    Returns:
        Tuple of:
            - cluster_labels: (H, W) array with cluster indices [0, n_clusters - 1].
            - cluster_centers_lab: (n_clusters, 3) true cluster centroid coordinates in CIELAB space.
    """
    if not isinstance(lab_image, np.ndarray) or lab_image.ndim != 3 or lab_image.shape[2] != 3:
        raise ValueError("lab_image must be a 3D numpy array with 3 channels.")

    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

    h, w, c = lab_image.shape
    features = extract_spectral_features(lab_image, bgr_image=bgr_image)

    # Use exact KMeans with k-means++ initialization for global convergence
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    labels_1d = kmeans.fit_predict(features)
    cluster_labels = labels_1d.reshape((h, w)).astype(np.int32)

    # Compute actual CIELAB centroids for semantic labeling
    raw_lab_flat = lab_image.reshape(-1, 3).astype(np.float32)
    cluster_centers_lab = np.zeros((n_clusters, 3), dtype=np.float32)

    for k in range(n_clusters):
        cluster_pixels = raw_lab_flat[labels_1d == k]
        if len(cluster_pixels) > 0:
            cluster_centers_lab[k] = np.mean(cluster_pixels, axis=0)
        else:
            cluster_centers_lab[k] = np.array([128.0, 128.0, 128.0], dtype=np.float32)

    return cluster_labels, cluster_centers_lab
