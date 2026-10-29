"""
Adaptive semantic land-cover labelling based on multi-spectral cluster properties, texture, and image contrast.
Maps K-means clusters to:
1: Dense Forest (deep green)
2: Farm Fields / Light Forest (green)
3: Water Bodies (blue)
4: Roads / Highways (black)
5: Buildings / Dry Land (yellow)
6: Unclassified (grey)

Incorporates:
- Multi-spectral CIELAB + Excess Green + Water Absorption Index (WAI)
- Local Texture Roughness (Standard Deviation) to separate glassy water from rough tree canopy
- Directional Line Detectors (DIP Lab Program 4) to capture thin, elongated road patterns
- Complete zero-unclassified adaptive resolution to eliminate orphaned grey pixels
"""
from typing import Dict, Tuple, List, Any, Optional
import cv2
import numpy as np
from backend.app.core.config import settings

CLASS_DENSE_FOREST = "Dense Forest"
CLASS_FARM_FIELDS = "Farm Fields / Light Forest"
CLASS_WATER = "Water Bodies"
CLASS_ROADS = "Roads / Highways"
CLASS_BUILDINGS_DRY_LAND = "Buildings / Dry Land"
CLASS_UNCLASSIFIED = "Unclassified"

# Directional Line Detector Kernels (DIP Lab Program 4: Line Detection)
_K_H = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
_K_V = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
_K_D1 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
_K_D2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

def analyze_centroid(lab_center: np.ndarray, texture_std: float = 10.0, line_response: float = 50.0) -> Dict[str, Any]:
    """
    Extract multi-spectral properties for a single CIELAB cluster center.
    """
    center_uint8 = np.clip(np.round(lab_center), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    rgb_arr = cv2.cvtColor(center_uint8, cv2.COLOR_LAB2RGB)
    bgr_arr = cv2.cvtColor(center_uint8, cv2.COLOR_LAB2BGR)
    hsv_arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
    
    rgb = rgb_arr[0, 0].astype(float)
    hsv = hsv_arr[0, 0].astype(float)
    
    L, a, b = lab_center
    r, g, blue = rgb
    hue, sat, val = hsv

    excess_green = 2.0 * g - r - blue
    wai = (blue - r) / (blue + r + 1e-4)

    return {
        "L": float(L),
        "a": float(a),
        "b": float(b),
        "r": float(r),
        "g": float(g),
        "blue": float(blue),
        "hue": float(hue),
        "sat": float(sat),
        "val": float(val),
        "exg": float(excess_green),
        "wai": float(wai),
        "texture_std": float(texture_std),
        "line_response": float(line_response),
    }


def classify_centroid_color(lab_center: np.ndarray) -> str:
    """
    Standalone heuristic classification for a single centroid.
    """
    p = analyze_centroid(lab_center)

    # 1. Water ("ultra smooth --> blue green color")
    is_cyan = (p["b"] < 127 and p["wai"] > 0.05 and p["blue"] > p["r"] * 1.2)
    is_blue = (p["blue"] > p["r"] * 1.05 and p["b"] < 126)
    is_dark_water = (p["L"] < 65 and p["r"] < 50 and p["g"] < 55 and p["blue"] < 60 and p["b"] <= 128)
    if (is_cyan or is_blue or is_dark_water) and p["b"] <= 130:
        return CLASS_WATER

    # 2. Vegetation
    is_green = (
        (p["exg"] >= 10 and p["g"] >= p["r"] * 0.95 and p["g"] > p["blue"])
        or (p["a"] < 124 and p["b"] >= 126 and p["g"] > p["blue"] * 0.9)
        or (25 <= p["hue"] <= 88 and p["sat"] >= 22 and p["g"] >= p["r"] * 0.92)
    )
    if is_green:
        return CLASS_DENSE_FOREST if (p["L"] < 105 or p["exg"] > 35) else CLASS_FARM_FIELDS

    # 3. Roads
    if (p["L"] < 85 or (p["r"] < 80 and p["g"] < 80 and p["blue"] < 80)) and p["sat"] < 32 and p["exg"] < 10:
        return CLASS_ROADS

    # 4. Buildings / Dry Land
    if p["b"] > 132 and p["r"] > 85 and p["exg"] < 12:
        return CLASS_BUILDINGS_DRY_LAND
    if p["L"] > 135 and p["sat"] < 40 and p["exg"] < 15:
        return CLASS_BUILDINGS_DRY_LAND

    # 5. Adaptive resolution
    if p["exg"] > 2 and p["g"] >= p["r"] * 0.92:
        return CLASS_FARM_FIELDS
    return CLASS_BUILDINGS_DRY_LAND


def assign_land_cover_labels(
    cluster_labels_2d: np.ndarray,
    cluster_centers_lab: np.ndarray,
    bgr_image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, str], np.ndarray]:
    """
    Assign land-cover semantic labels to K-Means clusters using texture roughness,
    directional line pattern responses, and adaptive contrast heuristics.
    """
    n_clusters = len(cluster_centers_lab)
    
    texture_map = None
    line_map = None
    if bgr_image is not None:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Texture roughness (local standard deviation)
        ksize = 9
        mean_int = cv2.blur(gray, (ksize, ksize))
        mean_sq = cv2.blur(gray * gray, (ksize, ksize))
        texture_map = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int))
        
        # Directional line pattern response
        r_h = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_H))
        r_v = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_V))
        r_d1 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_D1))
        r_d2 = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K_D2))
        line_map = np.maximum.reduce([r_h, r_v, r_d1, r_d2])

    centroids_info: List[Dict[str, Any]] = []
    for k in range(n_clusters):
        mask_k = (cluster_labels_2d == k)
        if bgr_image is not None and np.sum(mask_k) > 0:
            b_m = float(np.mean(bgr_image[:, :, 0][mask_k]))
            g_m = float(np.mean(bgr_image[:, :, 1][mask_k]))
            r_m = float(np.mean(bgr_image[:, :, 2][mask_k]))
            tex_m = float(np.mean(texture_map[mask_k])) if texture_map is not None else 10.0
            line_m = float(np.mean(line_map[mask_k])) if line_map is not None else 50.0

            center_uint8 = np.uint8([[[b_m, g_m, r_m]]])
            lab_arr = cv2.cvtColor(center_uint8, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
            hsv_arr = cv2.cvtColor(center_uint8, cv2.COLOR_BGR2HSV)[0, 0].astype(float)

            exg = 2.0 * g_m - r_m - b_m
            wai = (b_m - r_m) / (b_m + r_m + 1e-4)

            centroids_info.append({
                "empty": False,
                "L": float(lab_arr[0]),
                "a": float(lab_arr[1]),
                "b": float(lab_arr[2]),
                "r": float(r_m),
                "g": float(g_m),
                "blue": float(b_m),
                "hue": float(hsv_arr[0]),
                "sat": float(hsv_arr[1]),
                "val": float(hsv_arr[2]),
                "exg": float(exg),
                "wai": float(wai),
                "texture_std": float(tex_m),
                "line_response": float(line_m),
            })
        else:
            centroids_info.append(analyze_centroid(cluster_centers_lab[k]))

    # Scene-wide dynamic bounds
    all_L = [c["L"] for c in centroids_info]
    all_tex = [c["texture_std"] for c in centroids_info]
    min_L = min(all_L)
    max_L = max(all_L)
    min_tex = min(all_tex)

    cluster_to_class: Dict[int, str] = {}
    veg_cluster_indices = []
    non_veg_indices = []

    # Step 1: Detect Water Bodies ("water ultra smooth --> blue green color")
    for k in range(n_clusters):
        c = centroids_info[k]
        
        # Strict rejection: warm yellowish hue, red-dominance, or rough canopy
        is_rejected = (c["b"] > 132) or (c["r"] > c["blue"] * 1.18 and c["L"] > 68) or (c["texture_std"] > 16.0)
        
        # Cyan / turquoise mineral lakes
        is_cyan_lake = (c["b"] < 127 and c["wai"] > 0.05 and c["blue"] > c["r"] * 1.25)
        # Deep blue water
        is_blue_water = (c["b"] < 126 and (c["blue"] > c["r"] * 1.04 or c["wai"] > 0.0))
        # Deep dark water / ocean
        is_deep_dark = (c["L"] < 65 and c["r"] < 52 and c["g"] < 58 and c["blue"] < 62 and c["b"] <= 128)
        # Specular ultra-smooth glassy water
        is_ultra_smooth = (c["texture_std"] <= min_tex + 2.0 and c["texture_std"] < 6.5 and c["b"] <= 127 and c["wai"] > -0.05)

        if (is_cyan_lake or is_blue_water or is_deep_dark or is_ultra_smooth) and not is_rejected:
            cluster_to_class[k] = CLASS_WATER

    # Step 2: Identify Vegetation candidates
    for k in range(n_clusters):
        if k in cluster_to_class:
            continue
        c = centroids_info[k]
        is_veg = (
            (c["exg"] >= 12 and c["g"] >= c["r"])
            or (c["exg"] >= 8 and c["g"] > c["r"] * 0.98 and c["g"] > c["blue"] and c["sat"] >= 30)
            or (c["a"] < 122 and c["b"] >= 127 and c["g"] > c["r"] and c["exg"] > 6)
        )
        if is_veg:
            veg_cluster_indices.append(k)
        else:
            non_veg_indices.append(k)

    # Step 3: Classify Vegetation
    # Deep Forest: "rough, green" (canopy roughness, high ExG, lower lightness)
    # Farm Fields / Light Forest: "mild smooth if green" (lower roughness, moderate ExG)
    if len(veg_cluster_indices) == 1:
        k = veg_cluster_indices[0]
        c = centroids_info[k]
        is_dense = (c["L"] < 100 or c["exg"] > 32 or c["texture_std"] >= 12.5)
        cluster_to_class[k] = CLASS_DENSE_FOREST if is_dense else CLASS_FARM_FIELDS
    elif len(veg_cluster_indices) > 1:
        veg_scores = [
            (centroids_info[k]["exg"] * 1.5 - centroids_info[k]["L"] + centroids_info[k]["texture_std"] * 0.6)
            for k in veg_cluster_indices
        ]
        median_score = float(np.median(veg_scores))
        for k in veg_cluster_indices:
            score = centroids_info[k]["exg"] * 1.5 - centroids_info[k]["L"] + centroids_info[k]["texture_std"] * 0.6
            if score >= median_score:
                cluster_to_class[k] = CLASS_DENSE_FOREST
            else:
                cluster_to_class[k] = CLASS_FARM_FIELDS

    # Step 4: Classify Non-Vegetation
    # Roads: "always creates lines thin and smooth"
    # Buildings / Dry Land: "if yellow or white etc building / dry land"
    for k in non_veg_indices:
        c = centroids_info[k]
        
        # Road signature: high directional line response or low-saturation linear road corridor
        is_road_line = (c["line_response"] > 90 and c["sat"] < 35 and c["exg"] < 10 and c["L"] < 140)
        is_dark_road = (c["L"] < 85 or c["L"] <= min_L + 18) and c["sat"] < 35 and c["exg"] < 10
        is_medium_road = (
            c["L"] < 135
            and c["sat"] < 32
            and abs(c["a"] - 128) < 7
            and c["exg"] < 8
            and (max_L > 145 or c["line_response"] > 80)
        )
        if is_road_line or is_dark_road or is_medium_road:
            cluster_to_class[k] = CLASS_ROADS
            continue

        # Buildings / Dry Land: white concrete or yellow/warm soil, terracotta
        is_bright = (c["L"] >= 135) or (c["L"] >= max_L - 25 and c["exg"] < 15)
        is_dry_soil = (c["b"] > 130 and c["r"] > 80 and c["exg"] < 12)
        is_terracotta = (c["r"] > c["g"] * 1.05 and c["r"] > c["blue"] * 1.05 and (c["hue"] < 25 or c["hue"] > 160))

        if is_bright or is_dry_soil or is_terracotta:
            cluster_to_class[k] = CLASS_BUILDINGS_DRY_LAND
        else:
            # Step 5: Complete zero-unclassified adaptive resolution
            # If mildly green -> Farm Fields
            if c["exg"] > 2 and c["g"] >= c["r"] * 0.92:
                cluster_to_class[k] = CLASS_FARM_FIELDS
            elif c["L"] < 105 and c["sat"] < 30:
                cluster_to_class[k] = CLASS_ROADS
            else:
                cluster_to_class[k] = CLASS_BUILDINGS_DRY_LAND

    # Assign IDs and RGB colors
    cluster_to_id: Dict[int, int] = {}
    cluster_to_color: Dict[int, Tuple[int, int, int]] = {}

    for k in range(n_clusters):
        class_name = cluster_to_class.get(k, CLASS_BUILDINGS_DRY_LAND)
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
