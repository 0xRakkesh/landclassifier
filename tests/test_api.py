import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "app" in data

def test_classify_valid_image(client):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (34, 139, 34)
    _, encoded = cv2.imencode(".png", img)
    
    files = {"file": ("test.png", encoded.tobytes(), "image/png")}
    response = client.post("/classify?n_clusters=3", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "classes" in data
    assert "segmented_image_base64" in data
    assert data["parameters"]["n_clusters"] == 3

def test_classify_empty_file(client):
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/classify", files=files)
    assert response.status_code == 400

def test_classify_corrupted_file(client):
    files = {"file": ("corrupt.png", b"not-an-image", "image/png")}
    response = client.post("/classify", files=files)
    assert response.status_code == 400

def test_classify_invalid_even_kernel(client):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    files = {"file": ("test.png", encoded.tobytes(), "image/png")}
    response = client.post("/classify?blur_kernel=4", files=files)
    assert response.status_code == 400
