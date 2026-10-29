import numpy as np
import pytest
from backend.app.processing.refinement import refine_mask, generate_colored_mask
from backend.app.processing.metrics import calculate_area_statistics, compute_segmentation_metrics

def test_refine_mask():
    # 50x50 mask mostly class 1 (Dense Forest) with isolated class 2 (Farm Fields) noise pixel
    mask = np.ones((50, 50), dtype=np.int32)
    mask[25, 25] = 2 # Single isolated pixel
    
    refined = refine_mask(mask, kernel_size=3)
    assert refined.shape == (50, 50)
    # The isolated noise pixel should be cleaned by morphological filter
    assert refined[25, 25] == 1

def test_calculate_area_statistics():
    # 100x100 mask: 5000 pixels class 1, 5000 pixels class 2
    mask = np.ones((100, 100), dtype=np.int32)
    mask[50:, :] = 2
    
    stats = calculate_area_statistics(mask)
    assert len(stats) == 6 # 6 classes defined
    c1_stat = next(s for s in stats if s["class_id"] == 1)
    c2_stat = next(s for s in stats if s["class_id"] == 2)
    
    assert c1_stat["pixel_count"] == 5000
    assert c1_stat["percentage"] == 50.0
    assert c2_stat["pixel_count"] == 5000
    assert c2_stat["percentage"] == 50.0

def test_compute_segmentation_metrics():
    # Perfect match test
    gt = np.array([
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 4]
    ], dtype=np.int32)
    
    pred = gt.copy()
    metrics = compute_segmentation_metrics(pred, gt)
    
    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["macro_f1"] == 1.0
    
    # 1 error
    pred_with_err = gt.copy()
    pred_with_err[0, 0] = 2
    metrics_err = compute_segmentation_metrics(pred_with_err, gt)
    assert metrics_err["pixel_accuracy"] == round(8.0 / 9.0, 4)
    assert metrics_err["mean_iou"] < 1.0
