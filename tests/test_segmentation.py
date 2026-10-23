import numpy as np
import pytest
from backend.app.processing.segmentation import segment_kmeans, segment_dbscan
from backend.app.processing.labeling import assign_land_cover_labels, classify_centroid_color

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

def test_segment_dbscan_clustering():
    # Synthetic 2-region image
    lab = np.zeros((64, 64, 3), dtype=np.uint8)
    lab[:32, :] = [200, 100, 150] # Region 1
    lab[32:, :] = [50, 128, 120]  # Region 2
    
    labels, centers = segment_dbscan(lab, eps=10.0, min_samples=15)
    assert labels.shape == (64, 64)
    assert len(centers) >= 1
    assert np.all(labels >= 0)

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
    assert all(c in {"Vegetation", "Water", "Built-up", "Bare soil", "Other / uncertain"} for c in cluster_to_class.values())
