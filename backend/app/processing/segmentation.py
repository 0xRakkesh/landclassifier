"""
High-accuracy K-Means image segmentation module.
Combines:
1. Multi-spectral spectral feature extraction (CIELAB, Normalized Excess Green, and Water Index).
2. Spatial domain Sobel edge gradient magnitude (DIP Lab edge detection).
3. k-means++ centroid initialization for rapid and globally accurate convergence.
4. Pure NumPy SimpleKMeans implementation (ML Lab algorithm from scratch).
"""
from typing import Tuple, Optional
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SimpleKMeans:
    """
    Standard K-Means Clustering Algorithm from scratch (Machine Learning Lab implementation).
    
    Mathematical Formulation:
        Objective (Inertia / WCSS):
            J = sum_{k=1}^K sum_{x_i in S_k} || x_i - mu_k ||^2
            
        1. Assignment Step:
            c_i = argmin_k || x_i - mu_k ||^2
            
        2. Update Step:
            mu_k = (1 / |S_k|) * sum_{x_i in S_k} x_i
    """
    def __init__(self, n_clusters: int = 5, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = 0.0

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit SimpleKMeans on feature matrix X and return cluster label for each sample.
        
        Args:
            X: (N, D) float array of samples and features.
            
        Returns:
            (N,) integer array of cluster assignments [0, n_clusters - 1].
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(f"Number of samples ({n_samples}) must be >= n_clusters ({self.n_clusters})")

        # 1. k-means++ initialization for fast convergence and global accuracy
        centroids = np.empty((self.n_clusters, n_features), dtype=np.float32)
        first_idx = rng.randint(0, n_samples)
        centroids[0] = X[first_idx]

        closest_dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)
        for k in range(1, self.n_clusters):
            probs = closest_dist_sq / (np.sum(closest_dist_sq) + 1e-12)
            next_idx = rng.choice(n_samples, p=probs)
            centroids[k] = X[next_idx]
            dist_sq_k = np.sum((X - centroids[k]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_k)

        # 2. Lloyd's iterative alternating optimization
        labels = np.zeros(n_samples, dtype=np.int32)
        for iteration in range(self.max_iter):
            # Assignment Step: compute pairwise squared Euclidean distances
            dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2) ** 2
            new_labels = np.argmin(dists, axis=1).astype(np.int32)

            # Update Step: calculate new cluster means
            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                cluster_pts = X[new_labels == k]
                if len(cluster_pts) > 0:
                    new_centroids[k] = np.mean(cluster_pts, axis=0)
                else:
                    new_centroids[k] = X[rng.randint(0, n_samples)]

            # Check convergence tolerance
            centroid_shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            labels = new_labels

            if centroid_shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        dists = np.sum((X - centroids[labels]) ** 2)
        self.inertia_ = float(dists)

        return labels


def extract_spectral_features(
    lab_image: np.ndarray,
    bgr_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract multi-dimensional spectral + spatial feature representation per pixel:
    1. L* (Lightness in CIELAB)
    2. a* (Green-Red chromaticity in CIELAB)
    3. b* (Blue-Yellow chromaticity in CIELAB)
    4. ExG (Normalized Excess Green Index = 2G - R - B)
    5. WAI (Water Absorption Index = (B - R) / (B + R + eps))
    6. Edge (Sobel Gradient Magnitude from DIP Lab)
    
    Returns:
        (H*W, 6) standardized feature matrix.
    """
    h, w, c = lab_image.shape

    if bgr_image is None:
        bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    b = bgr_image[:, :, 0].astype(np.float32)
    g = bgr_image[:, :, 1].astype(np.float32)
    r = bgr_image[:, :, 2].astype(np.float32)

    L = lab_image[:, :, 0].astype(np.float32)
    a = lab_image[:, :, 1].astype(np.float32)
    lab_b = lab_image[:, :, 2].astype(np.float32)

    # 1. Excess Green Index: separates vegetation from built-up and water
    exg = (2.0 * g - r - b) / 255.0

    # 2. Water Absorption Index: separates water and dark regions
    wai = (b - r) / (b + r + 1e-4)

    # 3. Sobel Edge Gradient Magnitude (DIP Lab Program 3 & 11)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y) / 255.0

    # 4. Local Texture Roughness (Standard Deviation - DIP Lab Program 6 & 8)
    # Glassy smooth water has near-zero roughness (sigma < 4), while canopy/forest has rough texture (sigma > 12)
    ksize = 9
    mean_int = cv2.blur(gray, (ksize, ksize))
    mean_sq = cv2.blur(gray * gray, (ksize, ksize))
    texture_roughness = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int)) / 255.0

    # Combine into (H*W, 7) matrix
    features = np.column_stack([
        L.reshape(-1),
        a.reshape(-1),
        lab_b.reshape(-1),
        exg.reshape(-1),
        wai.reshape(-1),
        edge_mag.reshape(-1),
        texture_roughness.reshape(-1),
    ])

    # Standardize features (zero mean, unit variance) so K-Means weights all features fairly
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features


def segment_kmeans(
    lab_image: np.ndarray,
    bgr_image: Optional[np.ndarray] = None,
    n_clusters: int = 5,
    random_state: int = 42,
    use_simple_kmeans: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment satellite image using K-Means with k-means++ seeding.
    Supports both ML Lab SimpleKMeans and scikit-learn KMeans.
    
    Args:
        lab_image: (H, W, 3) CIELAB format image array.
        bgr_image: Optional (H, W, 3) BGR format image array.
        n_clusters: Number of clusters (typically 3, 5, or 7).
        random_state: Random seed for exact reproducibility.
        use_simple_kmeans: If True, uses the educational scratch SimpleKMeans implementation.
        
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

    if use_simple_kmeans:
        km = SimpleKMeans(n_clusters=n_clusters, max_iter=100, random_state=random_state)
        labels_1d = km.fit_predict(features)
    else:
        # scikit-learn KMeans with k-means++ seeding
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
