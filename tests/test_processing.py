import cv2
import numpy as np
import pytest
from backend.app.services.classifier import LandClassifierService

def test_classifier_service_end_to_end():
    # Create synthetic RGB satellite tile
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Green quadrant (Vegetation)
    img[:64, :64] = (30, 180, 40)
    # Blue quadrant (Water)
    img[:64, 64:] = (200, 80, 20) # BGR: High B, low R
    # Yellow-brown quadrant (Soil)
    img[64:, :64] = (30, 140, 180) # BGR: R=180, G=140, B=30
    # Gray quadrant (Built-up)
    img[64:, 64:] = (110, 110, 110)
    
    _, encoded = cv2.imencode(".png", img)
    raw_bytes = encoded.tobytes()

    result = LandClassifierService.classify(
        image_bytes=raw_bytes,
        n_clusters=5,
        apply_blur=True,
        blur_kernel=3,
        refine=True,
    )

    assert result["status"] == "success"
    assert result["processing_time_ms"] > 0
    assert "segmented_image_base64" in result
    assert "original_image_base64" in result
    assert len(result["classes"]) == 6
    
    # Check that area percentages sum to ~100%
    total_pct = sum(c["percentage"] for c in result["classes"])
    assert 99.0 <= total_pct <= 101.0
