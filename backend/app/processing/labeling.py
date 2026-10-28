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
    Classify a cluster centroid using physically grounded remote-sensing spectral characteristics:
    
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

    ex_green = 2 * g - r - blue

    # 1. Vegetation:
    # Distinct green component (positive excess green, green greater than red, and green hue)
    if (g > r * 1.03 and g > blue * 0.98 and ex_green > 6) or (
        30 <= hue <= 86 and sat > 30 and g >= r * 0.95
    ):
        return CLASS_VEGETATION

    # 2. Water bodies:
    # Pure water has very high absorption (deep water: L < 32 and dark RGB),
    # or clean open water with distinct blue dominance (blue >> red and blue >> green)
    if (L < 32 and val < 38) or (blue > r * 1.35 and blue > g * 1.12 and L < 80 and sat > 45):
        return CLASS_WATER

    # 3. Bare soil / Sand / Dry fields:
    # Warm earthy tones (R > B, G > B, yellow chromaticity b* > 134, warm hue 10..38)
    if (b > 134 and r > blue * 1.15 and g > blue * 1.02 and 10 <= hue <= 38) or (
        L > 122 and b > 138 and r >= g * 0.88 and r > blue
    ):
        return CLASS_BARE_SOIL

    # 4. Built-up (Urban infrastructure, concrete, asphalt roads, rooftops, railway tracks):
    # Concrete and asphalt have low-to-moderate saturation or neutral chromaticity across wide lightness
    if sat < 45 or (35 <= L <= 235 and abs(r - g) < 25 and abs(g - blue) < 25):
        return CLASS_BUILT_UP
    if (hue < 18 or hue > 160) and r > 115: # Red roof tiles / brick
        return CLASS_BUILT_UP

    # 5. Secondary fallbacks
    if ex_green > 10 and g > r:
        return CLASS_VEGETATION
    if L < 36 and blue >= r and blue >= g:
        return CLASS_WATER
    if b > 135 and r >= blue:
        return CLASS_BARE_SOIL
    if sat < 50:
        return CLASS_BUILT_UP

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
