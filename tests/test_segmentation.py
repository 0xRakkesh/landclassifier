import numpy as np
import pytest
from backend.app.processing.segmentation import segment_kmeans, extract_spectral_features, SimpleKMeans
from backend.app.processing.labeling import assign_land_cover_labels, classify_centroid_color

def test_extract_spectral_features():
    lab = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    features = extract_spectral_features(lab, bgr)
    assert features.shape == (100, 7) # (H*W, 7 features including texture roughness and Sobel edge magnitude)

def test_simple_kmeans_from_scratch():
    # Test ML Lab simple kmeans implementation
    X = np.array([
        [1.0, 1.0], [1.2, 0.9], [0.8, 1.1],
        [8.0, 8.0], [8.1, 7.9], [7.9, 8.2],
    ], dtype=np.float32)
    km = SimpleKMeans(n_clusters=2, random_state=42)
    labels = km.fit_predict(X)
    assert len(labels) == 6
    # Samples in cluster 1 should share a label, cluster 2 should share another
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert km.inertia_ > 0

def test_segment_kmeans_clusters():
    # Synthetic LAB image 64x64
    lab = np.zeros((64, 64, 3), dtype=np.uint8)
    lab[:32, :32] = [200, 100, 150] # Region 1
    lab[32:, 32:] = [50, 128, 120]  # Region 2
    lab[32:, :32] = [120, 140, 160] # Region 3
    
    labels, centers = segment_kmeans(lab, n_clusters=3, random_state=42)
    assert labels.shape == (64, 64)
    assert len(centers) == 3
    assert set(np.unique(labels)).issubset({0, 1, 2})

def test_assign_land_cover_labels():
    labels_2d = np.array([[0, 1], [2, 0]], dtype=np.int32)
    centers = np.array([
        [100.0, 90.0, 140.0],
        [30.0, 128.0, 120.0],
        [120.0, 135.0, 155.0]
    ], dtype=np.float32)

    class_mask, cluster_to_class, colored_rgb = assign_land_cover_labels(labels_2d, centers)
    assert class_mask.shape == (2, 2)
    assert colored_rgb.shape == (2, 2, 3)
    assert len(cluster_to_class) == 3
    valid_classes = {"Dense Forest", "Farm Fields / Light Forest", "Water Bodies", "Roads / Highways", "Buildings / Dry Land", "Unclassified"}
    assert all(c in valid_classes for c in cluster_to_class.values())
