import numpy as np
import pytest
from backend.app.processing.segmentation import segment_kmeans
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

def test_assign_land_cover_labels():
    labels_2d = np.array([[0, 1], [2, 0]], dtype=np.int32)
    # 3 centroids in LAB:
    # 0: Greenish (Vegetation) -> L=100, a=90, b=140
    # 1: Dark (Water) -> L=30, a=128, b=120
    # 2: Yellow-earth (Bare soil) -> L=120, a=135, b=155
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
