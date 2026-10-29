"""
Supervised Random Forest Land Classifier:
Calibrated directly on 550,000+ ground truth labeled pixels from LandCover.ai orthophotos.
Predicts:
- Class 0: Farm Fields / Light Forest (#22C55E Green)
- Class 1: Buildings / Dry Land (#FFD700 Yellow)
- Class 2: Dense Forest (#006400 Deep Green)
- Class 3: Water Bodies (#1E90FF Blue)
- Class 4: Roads / Highways (#141414 Black)
"""
import os
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
import joblib
from backend.app.core.config import settings
from backend.app.processing.refinement import refine_mask, generate_colored_mask
from backend.app.processing.metrics import calculate_area_statistics

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_landcover.joblib")

_MODEL_CACHE = None

# Directional Line Detector Kernels (DIP Lab Program 4)
_K_H = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
_K_V = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
_K_D1 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
_K_D2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

# Mapping from LandCover.ai Ground Truth class IDs (0..4) to Application Class IDs (1..5)
# settings.CLASSES:
# 1: Dense Forest
# 2: Farm Fields / Light Forest
# 3: Water Bodies
# 4: Roads / Highways
# 5: Buildings / Dry Land
GT_TO_APP_ID = {
    0: 2, # Fields -> ID 2
    1: 5, # Buildings -> ID 5 (Yellow)
    2: 1, # Dense Forest -> ID 1
    3: 3, # Water Bodies -> ID 3
    4: 5, # Roads -> ID 5 (Yellow - merged into same class as Buildings & Dry Land)
}

def get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            remote_url = os.environ.get("MODEL_DOWNLOAD_URL")
            if remote_url:
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                import urllib.request
                urllib.request.urlretrieve(remote_url, MODEL_PATH)
            else:
                raise FileNotFoundError(
                    f"Trained Random Forest model not found at {MODEL_PATH}. "
                    "For cloud deployment, upload the model or provide MODEL_DOWNLOAD_URL."
                )
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE

def extract_features(bgr_img: np.ndarray) -> np.ndarray:
    """Extract 8 multi-spectral, textural, and geometric features."""
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    b = bgr_img[:, :, 0].astype(np.float32)
    g = bgr_img[:, :, 1].astype(np.float32)
    r = bgr_img[:, :, 2].astype(np.float32)

    L = lab_img[:, :, 0].astype(np.float32)
    a = lab_img[:, :, 1].astype(np.float32)
    lab_b = lab_img[:, :, 2].astype(np.float32)

    exg = (2.0 * g - r - b) / 255.0
    wai = (b - r) / (b + r + 1e-4)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y) / 255.0

    mean_int = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    texture_roughness = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int)) / 255.0

    r_h = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_H))
    r_v = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_V))
    r_d1 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_D1))
    r_d2 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_D2))
    line_response = np.maximum.reduce([r_h, r_v, r_d1, r_d2]) / 255.0

    features = np.column_stack([
        L.reshape(-1),
        a.reshape(-1),
        lab_b.reshape(-1),
        exg.reshape(-1),
        wai.reshape(-1),
        edge_mag.reshape(-1),
        texture_roughness.reshape(-1),
        line_response.reshape(-1),
    ])
    return features

def classify_random_forest(
    bgr_image: np.ndarray,
    refine: bool = True,
    refine_kernel: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Run Supervised Random Forest classification on preprocessed BGR image.
    
    Returns:
        Tuple of (final_class_mask_2d, colored_mask_rgb, area_statistics)
    """
    payload = get_model()
    model = payload["model"]

    h, w = bgr_image.shape[:2]
    features = extract_features(bgr_image)

    # Predict ground truth class indices (0..4)
    raw_preds_1d = model.predict(features)
    raw_mask_2d = raw_preds_1d.reshape((h, w)).astype(np.int32)

    # Map dataset IDs (0..4) to Application class IDs (1..5)
    app_mask_2d = np.zeros((h, w), dtype=np.int32)
    for gt_id, app_id in GT_TO_APP_ID.items():
        app_mask_2d[raw_mask_2d == gt_id] = app_id

    # Optional morphological refinement
    if refine:
        final_class_mask = refine_mask(app_mask_2d, kernel_size=refine_kernel)
    else:
        final_class_mask = app_mask_2d

    colored_mask_rgb = generate_colored_mask(final_class_mask)
    area_stats = calculate_area_statistics(final_class_mask)

    return final_class_mask, colored_mask_rgb, area_stats
