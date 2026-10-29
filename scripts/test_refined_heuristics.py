import sys
import os
import glob
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.core.config import settings

DATASET_DIR = r"c:\Users\rakes\OneDrive\Desktop\landcover.ai.v1"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")

# Directional Line Detector Kernels (DIP Lab Program 4)
k_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
k_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
k_d1 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
k_d2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

def extract_features_and_profile(bgr_img, lab_img, cluster_labels_2d, n_clusters):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 1. Texture Roughness (9x9 local std)
    mean_int = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    texture_map = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int))
    
    # 2. Line Response (thin, linear patterns)
    r_h = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_h))
    r_v = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_v))
    r_d1 = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_d1))
    r_d2 = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_d2))
    line_map = np.maximum.reduce([r_h, r_v, r_d1, r_d2])

    centroids_info = []
    for k in range(n_clusters):
        mask_k = (cluster_labels_2d == k)
        if np.sum(mask_k) == 0:
            centroids_info.append({"empty": True})
            continue

        b_m = float(np.mean(bgr_img[:, :, 0][mask_k]))
        g_m = float(np.mean(bgr_img[:, :, 1][mask_k]))
        r_m = float(np.mean(bgr_img[:, :, 2][mask_k]))
        
        L_m = float(np.mean(lab_img[:, :, 0][mask_k]))
        a_m = float(np.mean(lab_img[:, :, 1][mask_k]))
        b_lab_m = float(np.mean(lab_img[:, :, 2][mask_k]))
        
        tex_m = float(np.mean(texture_map[mask_k]))
        line_m = float(np.mean(line_map[mask_k]))

        center_uint8 = np.uint8([[[b_m, g_m, r_m]]])
        hsv_arr = cv2.cvtColor(center_uint8, cv2.COLOR_BGR2HSV)[0, 0].astype(float)

        exg = 2.0 * g_m - r_m - b_m
        wai = (b_m - r_m) / (b_m + r_m + 1e-4)

        centroids_info.append({
            "empty": False,
            "L": L_m, "a": a_m, "b": b_lab_m,
            "r": r_m, "g": g_m, "blue": b_m,
            "hue": float(hsv_arr[0]), "sat": float(hsv_arr[1]), "val": float(hsv_arr[2]),
            "exg": float(exg), "wai": float(wai),
            "texture_std": float(tex_m),
            "line_response": float(line_m),
            "pct": float(np.sum(mask_k) / cluster_labels_2d.size * 100.0)
        })

    return centroids_info

def classify_clusters_refined(centroids_info, n_clusters):
    cluster_to_class = {}
    
    valid_c = [c for c in centroids_info if not c.get("empty", False)]
    if not valid_c:
        return {k: "Unclassified" for k in range(n_clusters)}

    all_L = [c["L"] for c in valid_c]
    all_tex = [c["texture_std"] for c in valid_c]
    min_L = min(all_L)
    max_L = max(all_L)
    min_tex = min(all_tex)

    # Step 1: Water Detection
    # "water ultra smooth --> blue green color"
    for k in range(n_clusters):
        c = centroids_info[k]
        if c.get("empty", False):
            continue

        # Rejection: warm yellowish hue or red dominance or high canopy roughness
        is_rejected = (c["b"] > 132) or (c["r"] > c["blue"] * 1.18 and c["L"] > 68) or (c["texture_std"] > 16.0)
        
        # Cyan / turquoise mineral lakes
        is_cyan = (c["b"] < 127 and c["wai"] > 0.05 and c["blue"] > c["r"] * 1.25)
        # Deep blue water
        is_blue = (c["b"] < 126 and (c["blue"] > c["r"] * 1.04 or c["wai"] > 0.0))
        # Deep dark water
        is_dark_water = (c["L"] < 65 and c["r"] < 52 and c["g"] < 58 and c["blue"] < 62 and c["b"] <= 128)
        # Ultra smooth glassy water
        is_ultra_smooth = (c["texture_std"] <= min_tex + 2.0 and c["texture_std"] < 6.5 and c["b"] <= 127 and c["wai"] > -0.05)

        if (is_cyan or is_blue or is_dark_water or is_ultra_smooth) and not is_rejected:
            cluster_to_class[k] = "Water Bodies"

    # Step 2: Separate Vegetation Candidates
    veg_indices = []
    non_veg_indices = []
    for k in range(n_clusters):
        if k in cluster_to_class:
            continue
        c = centroids_info[k]
        if c.get("empty", False):
            continue

        is_veg = (
            (c["exg"] >= 12 and c["g"] >= c["r"])
            or (c["exg"] >= 8 and c["g"] > c["r"] * 0.98 and c["g"] > c["blue"] and c["sat"] >= 30)
            or (c["a"] < 122 and c["b"] >= 127 and c["g"] > c["r"] and c["exg"] > 6)
        )
        if is_veg:
            veg_indices.append(k)
        else:
            non_veg_indices.append(k)

    # Step 3: Deep Forest ("rough, green") vs Farm Fields / Light Forest ("mild smooth if green")
    if len(veg_indices) == 1:
        k = veg_indices[0]
        c = centroids_info[k]
        is_dense = (c["L"] < 100 or c["exg"] > 32 or c["texture_std"] >= 12.5)
        cluster_to_class[k] = "Dense Forest" if is_dense else "Farm Fields / Light Forest"
    elif len(veg_indices) > 1:
        # Score combines canopy roughness and excess green
        veg_scores = [
            (centroids_info[k]["exg"] * 1.5 - centroids_info[k]["L"] + centroids_info[k]["texture_std"] * 0.6)
            for k in veg_indices
        ]
        med_score = float(np.median(veg_scores))
        for k in veg_indices:
            score = centroids_info[k]["exg"] * 1.5 - centroids_info[k]["L"] + centroids_info[k]["texture_std"] * 0.6
            if score >= med_score:
                cluster_to_class[k] = "Dense Forest"
            else:
                cluster_to_class[k] = "Farm Fields / Light Forest"

    # Step 4: Roads ("always creates lines thin and smooth") vs Buildings / Dry Land ("yellow or white")
    for k in non_veg_indices:
        c = centroids_info[k]
        
        # Road signature: achromatic, line pattern, smooth surface
        is_road_line = (c["line_response"] > 95 and c["sat"] < 35 and c["exg"] < 10 and c["L"] < 140)
        is_dark_road = (c["L"] < 85 or c["L"] <= min_L + 18) and c["sat"] < 35 and c["exg"] < 10
        is_medium_road = (
            c["L"] < 135
            and c["sat"] < 32
            and abs(c["a"] - 128) < 7
            and c["exg"] < 8
            and (max_L > 145 or c["line_response"] > 80)
        )
        
        if is_road_line or is_dark_road or is_medium_road:
            cluster_to_class[k] = "Roads / Highways"
            continue

        # Buildings / Dry Land: yellow or white
        is_bright = (c["L"] >= 135) or (c["L"] >= max_L - 25 and c["exg"] < 15)
        is_dry_soil = (c["b"] > 130 and c["r"] > 80 and c["exg"] < 12)
        is_terracotta = (c["r"] > c["g"] * 1.05 and c["r"] > c["blue"] * 1.05 and (c["hue"] < 25 or c["hue"] > 160))

        if is_bright or is_dry_soil or is_terracotta:
            cluster_to_class[k] = "Buildings / Dry Land"
        else:
            # Step 5: Adaptive resolution for remaining (Eliminate Unclassified waste!)
            # If mildly green -> farm fields
            if c["exg"] > 2 and c["g"] >= c["r"] * 0.92:
                cluster_to_class[k] = "Farm Fields / Light Forest"
            elif c["L"] < 105 and c["sat"] < 30:
                cluster_to_class[k] = "Roads / Highways"
            else:
                cluster_to_class[k] = "Buildings / Dry Land"

    return cluster_to_class

def test_sample():
    img_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif"
    img = load_image(img_path)
    bgr, lab = preprocess_image(img, target_size=(512, 512))
    labels, centers = segment_kmeans(lab, bgr_image=bgr, n_clusters=5, random_state=42)
    info = extract_features_and_profile(bgr, lab, labels, 5)
    mapping = classify_clusters_refined(info, 5)
    print("M-34-51-C-b-2-1 Mapping:")
    for k, v in mapping.items():
        c = info[k]
        print(f"  Cluster {k} ({c['pct']:.1f}%): L={c['L']:.1f}, ExG={c['exg']:.1f}, Tex={c['texture_std']:.1f}, Line={c['line_response']:.1f} -> {v}")

if __name__ == "__main__":
    test_sample()
