"""
Heuristic semantic land-cover labelling based on cluster color characteristics.
Maps K-means clusters to:
1: Vegetation
2: Water
3: Built-up
4: Bare soil
5: Other / uncertain
"""
from typing import Dict, Tuple
import cv2
import numpy as np
from backend.app.core.config import settings

CLASS_VEGETATION = "Vegetation"
CLASS_WATER = "Water"
CLASS_BUILT_UP = "Built-up"
CLASS_BARE_SOIL = "Bare soil"
CLASS_UNCERTAIN = "Other / uncertain"

def classify_centroid_color(lab_center: np.ndarray) -> str:
    """
    Classify a cluster centroid based on CIELAB and derived RGB/HSV spectral characteristics.
    
    OpenCV CIELAB ranges:
    L: 0..255 (0 = black, 255 = white)
    a: 0..255 (128 = neutral, < 128 = green, > 128 = red/magenta)
    b: 0..255 (128 = neutral, < 128 = blue, > 128 = yellow)
    """
    center_uint8 = np.clip(np.round(lab_center), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    rgb_arr = cv2.cvtColor(center_uint8, cv2.COLOR_LAB2RGB)
    bgr_arr = cv2.cvtColor(center_uint8, cv2.COLOR_LAB2BGR)
    hsv_arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
    
    rgb = rgb_arr[0, 0].astype(float)
    hsv = hsv_arr[0, 0].astype(float)
    
    L, a, b = lab_center
    r, g, blue = rgb
    hue, sat, val = hsv  # Hue: 0..179, Sat: 0..255, Val: 0..255

    excess_green = 2 * g - r - blue

    # 1. Water bodies:
    # High optical absorption (low L < 45) or blue dominance (blue > r, cool hue 85-140)
    if L < 48 or (blue > r * 1.08 and blue >= g * 0.98) or (88 <= hue <= 140 and sat > 20 and L < 110):
        return CLASS_WATER

    # 2. Bare soil / Dry fields / Sand:
    # Warm earthy tones (high b* > 136 in LAB), moderate to high lightness, r and g both elevated
    if (b > 136 and r > 90 and g > 85 and r >= g * 0.90) or (L > 120 and b > 138):
        return CLASS_BARE_SOIL

    # 3. Built-up / Concrete / Asphalt:
    # Low saturation (neutral gray / road surfaces) or bright structural surfaces
    if (sat < 22 and 45 <= L <= 215) or (abs(r - g) < 8 and abs(g - blue) < 8 and sat < 28):
        return CLASS_BUILT_UP

    # 4. Vegetation:
    # Noticeable green chroma (a < 118 in LAB), excess green > 15, and green distinctly greater than red
    if (a < 118 and g > r * 1.05 and g > blue * 1.10 and excess_green > 12) or (
        35 <= hue <= 85 and sat > 35 and g > r * 1.05
    ):
        return CLASS_VEGETATION

    # 5. Secondary / Fallback heuristics
    if excess_green > 25 and g > r * 1.10:
        return CLASS_VEGETATION
    if L > 130 and b > 132 and r >= g * 0.88:
        return CLASS_BARE_SOIL
    if sat < 32 and 40 <= L <= 220:
        return CLASS_BUILT_UP
    if blue > r and blue > g:
        return CLASS_WATER

    # Ambiguous / mixed terrain
    return CLASS_UNCERTAIN


def assign_land_cover_labels(
    cluster_labels_2d: np.ndarray,
    cluster_centers_lab: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, str], np.ndarray]:
    """
    Assign land-cover semantic labels to K-Means clusters and produce colored segmentation map.
    
    Args:
        cluster_labels_2d: (H, W) array of cluster IDs [0, K-1].
        cluster_centers_lab: (K, 3) centroid positions in CIELAB space.
        
    Returns:
        Tuple of:
            - class_mask_2d: (H, W) array containing integer class IDs (1 to 5).
            - cluster_to_class: Dict mapping cluster ID -> Class Name string.
            - colored_mask_rgb: (H, W, 3) RGB visualization of the segmentation.
    """
    n_clusters = len(cluster_centers_lab)
    cluster_to_class: Dict[int, str] = {}
    cluster_to_id: Dict[int, int] = {}
    cluster_to_color: Dict[int, Tuple[int, int, int]] = {}

    for k in range(n_clusters):
        class_name = classify_centroid_color(cluster_centers_lab[k])
        cluster_to_class[k] = class_name
        class_info = settings.CLASSES[class_name]
        cluster_to_id[k] = class_info["id"]
        cluster_to_color[k] = class_info["color_rgb"]

    h, w = cluster_labels_2d.shape
    class_mask_2d = np.zeros((h, w), dtype=np.int32)
    colored_mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for k in range(n_clusters):
        mask_k = (cluster_labels_2d == k)
        class_mask_2d[mask_k] = cluster_to_id[k]
        colored_mask_rgb[mask_k] = cluster_to_color[k]

    return class_mask_2d, cluster_to_class, colored_mask_rgb
