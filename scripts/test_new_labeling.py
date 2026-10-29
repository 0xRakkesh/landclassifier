import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import analyze_centroid
from backend.app.core.config import settings

def test_new_labeling_logic(img_path):
    img = load_image(img_path)
    resized_bgr, lab_image = preprocess_image(
        img,
        target_size=settings.DEFAULT_IMAGE_SIZE,
        apply_blur=True,
        blur_kernel=5,
        use_bilateral=True,
    )

    cluster_labels_2d, cluster_centers_lab = segment_kmeans(
        lab_image,
        bgr_image=resized_bgr,
        n_clusters=5,
        random_state=42,
    )

    n_clusters = len(cluster_centers_lab)
    centroids = [analyze_centroid(cluster_centers_lab[k]) for k in range(n_clusters)]

    all_exg = [c["exg"] for c in centroids]
    all_L = [c["L"] for c in centroids]
    min_L = min(all_L)
    max_L = max(all_L)

    mapping = {}
    veg_indices = []
    non_veg_indices = []

    # Step 1: Detect Water
    # Supports both dark/deep water, clear blue water, and cyan/turquoise mineral/quarry lakes
    for k, c in enumerate(centroids):
        # Cyan / turquoise water: high blue & green, very low red, WAI > 0.05, b < 128
        is_cyan_lake = (c["b"] < 127 and c["wai"] > 0.05 and c["blue"] > c["r"] * 1.3)
        # Deep blue water: blue-dominant, cool axis
        is_blue_water = (c["b"] < 126 and (c["blue"] > c["r"] * 1.05 or c["wai"] > 0.0))
        # Deep dark water: very low luminance, low reflectance, non-vegetation (b <= 128)
        is_deep_dark = (c["L"] < 65 and c["r"] < 50 and c["g"] < 55 and c["blue"] < 60 and c["b"] <= 128)
        
        # Strict rejection for non-water: warm yellowish hue or high red-to-blue ratio
        is_rejected = (c["b"] > 132) or (c["r"] > c["blue"] * 1.15 and c["L"] > 70)
        
        if (is_cyan_lake or is_blue_water or is_deep_dark) and not is_rejected:
            mapping[k] = "Water Bodies"

    # Step 2: Separate Vegetation from Non-Vegetation among remaining
    for k, c in enumerate(centroids):
        if k in mapping:
            continue
        # Vegetation signature: positive ExG or green LAB coordinate
        is_veg = (
            (c["exg"] >= 10 and c["g"] >= c["r"] * 0.95 and c["g"] > c["blue"])
            or (c["a"] < 123 and c["b"] >= 126 and c["g"] > c["blue"])
            or (25 <= c["hue"] <= 85 and c["sat"] >= 25 and c["g"] >= c["r"] * 0.95)
        )
        if is_veg:
            veg_indices.append(k)
        else:
            non_veg_indices.append(k)

    # Step 3: Classify Vegetation clusters into Dense Forest vs Farm Fields / Light Forest
    if len(veg_indices) == 1:
        k = veg_indices[0]
        c = centroids[k]
        mapping[k] = "Dense Forest" if (c["L"] < 100 or c["exg"] > 35) else "Farm Fields / Light Forest"
    elif len(veg_indices) > 1:
        # Score vegetation by darkness and green intensity: Dense forest has higher ExG and lower L
        veg_scores = [(centroids[k]["exg"] * 1.5 - centroids[k]["L"]) for k in veg_indices]
        median_score = np.median(veg_scores)
        for k in veg_indices:
            score = (centroids[k]["exg"] * 1.5 - centroids[k]["L"])
            if score >= median_score:
                mapping[k] = "Dense Forest"
            else:
                mapping[k] = "Farm Fields / Light Forest"

    # Step 4: Classify Non-Vegetation (Roads vs Buildings / Dry Land)
    for k in non_veg_indices:
        c = centroids[k]
        # Dark surface or intermediate neutral road:
        is_dark_road = (c["L"] < 80 or c["L"] <= min_L + 15) and c["sat"] < 35 and c["exg"] < 10
        is_medium_road = (
            c["L"] < 135 
            and max_L > 150  # there is bright dry land in the scene
            and c["sat"] < 30
            and abs(c["a"] - 128) < 6
            and c["exg"] < 8
        )
        if is_dark_road or is_medium_road:
            mapping[k] = "Roads / Highways"
            continue

        # Bright structures, dry sand, bare soil
        is_bright = (c["L"] >= 135) or (c["L"] >= max_L - 25)
        is_dry_soil = (c["b"] > 132 and c["r"] > 85 and c["exg"] < 12)
        is_terracotta = (c["r"] > c["g"] * 1.05 and c["r"] > c["blue"] * 1.05 and (c["hue"] < 25 or c["hue"] > 160))

        if is_bright or is_dry_soil or is_terracotta:
            mapping[k] = "Buildings / Dry Land"
        else:
            mapping[k] = "Unclassified"

    print("\n--- Final Mapping ---")
    for k, v in mapping.items():
        c = centroids[k]
        pct = np.sum(cluster_labels_2d == k) / cluster_labels_2d.size * 100
        print(f"Cluster {k} ({pct:.1f}%): L={c['L']:.1f}, ExG={c['exg']:.1f}, WAI={c['wai']:.3f}, b={c['b']:.1f} -> {v}")

test_new_labeling_logic(r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif")
