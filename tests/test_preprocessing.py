import cv2
import numpy as np
import pytest
from backend.app.processing.preprocessing import load_image, preprocess_image

def test_load_image_from_bytes():
    # Create synthetic test BGR image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (0, 255, 0) # Green
    _, encoded = cv2.imencode(".png", img)
    
    loaded = load_image(encoded.tobytes())
    assert loaded.shape == (100, 100, 3)
    assert np.array_equal(loaded[0, 0], [0, 255, 0])

def test_load_image_grayscale_converted_to_3ch():
    gray = np.full((80, 80), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", gray)
    loaded = load_image(encoded.tobytes())
    assert loaded.shape == (80, 80, 3)
    assert loaded[0, 0, 0] == 128

def test_load_image_invalid_bytes():
    with pytest.raises(ValueError):
        load_image(b"not an image file")

def test_load_image_empty_bytes():
    with pytest.raises(ValueError):
        load_image(b"")

def test_preprocess_image_resizing_and_lab():
    # 200x150 BGR
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    img[:, :, 1] = 200  # greenish
    
    target_size = (300, 300)
    resized_bgr, lab_image = preprocess_image(img, target_size=target_size, apply_blur=True, blur_kernel=3)
    
    assert resized_bgr.shape == (300, 300, 3)
    assert lab_image.shape == (300, 300, 3)
    assert lab_image.dtype == np.uint8

def test_preprocess_image_invalid_kernel():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_image(img, apply_blur=True, blur_kernel=4) # Even kernel invalid
