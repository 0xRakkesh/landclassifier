import cv2
import numpy as np
import pytest
from backend.app.services.classifier import LandClassifierService
from backend.app.core.config import settings

def test_health_response_config():
    assert settings.PROJECT_NAME == "Land Classification System"
    assert settings.VERSION == "2.2.0"
    assert len(settings.CLASSES) == 6
    assert "Dense Forest" in settings.CLASSES
    assert "Farm Fields / Light Forest" in settings.CLASSES
    assert "Water Bodies" in settings.CLASSES
    assert "Roads / Highways" in settings.CLASSES
    assert "Buildings / Dry Land" in settings.CLASSES
    assert "Unclassified" in settings.CLASSES

def test_service_classify_valid_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (34, 139, 34)
    _, encoded = cv2.imencode(".png", img)
    
    file_bytes = encoded.tobytes()
    resp = LandClassifierService.classify(
        image_bytes=file_bytes,
        algorithm="kmeans",
        n_clusters=3,
        apply_blur=True,
        blur_kernel=3,
        use_bilateral=True,
        refine=True,
        refine_kernel=3,
    )
    assert resp["status"] == "success"
    assert resp["algorithm"] == "kmeans"
    assert len(resp["classes"]) == 6
    assert resp["parameters"]["n_clusters"] == 3

def test_service_classify_water_detection():
    # Test that blue water pixels are classified as Water Bodies
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (220, 80, 20) # BGR: High Blue, low Red
    _, encoded = cv2.imencode(".png", img)
    
    resp = LandClassifierService.classify(
        image_bytes=encoded.tobytes(),
        algorithm="kmeans",
        n_clusters=2,
    )
    water_stat = next(s for s in resp["classes"] if s["class_name"] == "Water Bodies")
    assert water_stat["percentage"] > 80.0

def test_service_classify_empty_file():
    with pytest.raises(ValueError):
        LandClassifierService.classify(image_bytes=b"")

def test_service_classify_invalid_even_kernel():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    with pytest.raises(ValueError):
        LandClassifierService.classify(image_bytes=encoded.tobytes(), blur_kernel=4)

def test_service_classify_random_forest():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (30, 180, 40)
    _, encoded = cv2.imencode(".png", img)
    resp = LandClassifierService.classify(
        image_bytes=encoded.tobytes(),
        algorithm="random_forest",
    )
    assert resp["status"] == "success"
    assert "Random Forest" in resp["algorithm"]
    assert len(resp["classes"]) == 6
