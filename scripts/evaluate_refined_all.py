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

k_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
k_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
k_d1 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
k_d2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

def extract_features_and_profile(bgr_img, lab_img, cluster_labels_2d, n_clusters):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_int = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    texture_map = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int))
    
    r_h = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_h))
    r_v = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_v))
    r_d1 = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_d1))
    r_d2 = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_d2))
    line_map = np.maximum.reduce([r_h, r_v, r_d1, r_d2])

    centroids_info = []
    for k in range(n_clusters):
        mask_k = (cluster_labels_2d == k)
        count = int(np.sum(mask_k))
        if count == 0:
            centroids_info.append({"empty": True, "count": 0, "pct": 0.0})
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
            "count": count,
            "L": L_m, "a": a_m, "b": b_lab_m,
            "r": r_m, "g": g_m, "blue": b_m,
            "hue": float(hsv_arr[0]), "sat": float(hsv_arr[1]), "val": float(hsv_arr[2]),
            "exg": float(exg), "wai": float(wai),
            "texture_std": float(tex_m),
            "line_response": float(line_m),
            "pct": float(count / cluster_labels_2d.size * 100.0)
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

    # 1. Water Detection: "water ultra smooth --> blue green color"
    for k in range(n_clusters):
        c = centroids_info[k]
        if c.get("empty", False):
            continue

        is_rejected = (c["b"] > 132) or (c["r"] > c["blue"] * 1.18 and c["L"] > 68) or (c["texture_std"] > 16.0)
        is_cyan = (c["b"] < 127 and c["wai"] > 0.05 and c["blue"] > c["r"] * 1.25)
        is_blue = (c["b"] < 126 and (c["blue"] > c["r"] * 1.04 or c["wai"] > 0.0))
        is_dark_water = (c["L"] < 65 and c["r"] < 52 and c["g"] < 58 and c["blue"] < 62 and c["b"] <= 128)
        is_ultra_smooth = (c["texture_std"] <= min_tex + 2.0 and c["texture_std"] < 6.5 and c["b"] <= 127 and c["wai"] > -0.05)

        if (is_cyan or is_blue or is_dark_water or is_ultra_smooth) and not is_rejected:
            cluster_to_class[k] = "Water Bodies"

    # 2. Vegetation
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

    # 3. Dense Forest ("rough, green") vs Farm Fields / Light Forest ("mild smooth if green")
    if len(veg_indices) == 1:
        k = veg_indices[0]
        c = centroids_info[k]
        is_dense = (c["L"] < 100 or c["exg"] > 32 or c["texture_std"] >= 12.5)
        cluster_to_class[k] = "Dense Forest" if is_dense else "Farm Fields / Light Forest"
    elif len(veg_indices) > 1:
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

    # 4. Roads ("lines thin and smooth") vs Buildings / Dry Land ("yellow or white")
    for k in non_veg_indices:
        c = centroids_info[k]
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
            cluster_to_class[k] = "Roads / Highways"
            continue

        is_bright = (c["L"] >= 135) or (c["L"] >= max_L - 25 and c["exg"] < 15)
        is_dry_soil = (c["b"] > 130 and c["r"] > 80 and c["exg"] < 12)
        is_terracotta = (c["r"] > c["g"] * 1.05 and c["r"] > c["blue"] * 1.05 and (c["hue"] < 25 or c["hue"] > 160))

        if is_bright or is_dry_soil or is_terracotta:
            cluster_to_class[k] = "Buildings / Dry Land"
        else:
            # Step 5: Adaptive resolution for remaining
            if c["exg"] > 2 and c["g"] >= c["r"] * 0.92:
                cluster_to_class[k] = "Farm Fields / Light Forest"
            elif c["L"] < 105 and c["sat"] < 30:
                cluster_to_class[k] = "Roads / Highways"
            else:
                cluster_to_class[k] = "Buildings / Dry Land"

    return cluster_to_class

def evaluate_all():
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))

    print(f"Testing refined heuristics across {len(img_paths)} images...")
    gt_total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    pred_total_counts = {
        "Dense Forest": 0,
        "Farm Fields / Light Forest": 0,
        "Water Bodies": 0,
        "Roads / Highways": 0,
        "Buildings / Dry Land": 0,
        "Unclassified": 0
    }

    t0 = time.time()
    for idx, (img_p, mask_p) in enumerate(zip(img_paths, mask_paths)):
        img = load_image(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        
        bgr, lab = preprocess_image(img, target_size=(512, 512))
        mask_r = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        labels, centers = segment_kmeans(lab, bgr_image=bgr, n_clusters=5, random_state=42)
        info = extract_features_and_profile(bgr, lab, labels, 5)
        mapping = classify_clusters_refined(info, 5)

        for gid in range(5):
            gt_total_counts[gid] += int(np.sum(mask_r == gid))

        for k in range(5):
            cls_name = mapping.get(k, "Unclassified")
            cnt = info[k]["count"]
            pred_total_counts[cls_name] += cnt

    total_gt = sum(gt_total_counts.values())
    total_pred = sum(pred_total_counts.values())
    elapsed = time.time() - t0

    print(f"\nEvaluated in {elapsed:.1f}s!")
    print("\n================ REFINED DATASET-WIDE MATRIX ================")
    print(f"{'Land-Cover Class':<28} | {'Pred Area %':<12} | {'LandCover.ai GT %':<18} | {'GT Class Reference'}")
    print("-" * 85)
    for cname in ["Water Bodies", "Dense Forest", "Farm Fields / Light Forest", "Roads / Highways", "Buildings / Dry Land", "Unclassified"]:
        p_pct = pred_total_counts[cname] / total_pred * 100.0
        gt_info = ""
        gt_pct = 0.0
        if cname == "Water Bodies":
            gt_pct = gt_total_counts[3] / total_gt * 100.0
            gt_info = "GT 3 (Water)"
        elif cname == "Dense Forest":
            gt_pct = gt_total_counts[2] / total_gt * 100.0
            gt_info = "GT 2 (Woodland / Forest)"
        elif cname == "Farm Fields / Light Forest":
            gt_info = "Part of GT 0 (Cropland / Grass)"
        elif cname == "Roads / Highways":
            gt_pct = gt_total_counts[4] / total_gt * 100.0
            gt_info = "GT 4 (Roads & Highways)"
        elif cname == "Buildings / Dry Land":
            gt_pct = gt_total_counts[1] / total_gt * 100.0
            gt_info = "GT 1 (Buildings) + Dry bare soil"
        elif cname == "Unclassified":
            gt_info = "Missing / Shadows"

        gt_str = f"{gt_pct:>16.2f}%" if gt_pct > 0 else f"{'--':>17}"
        print(f"{cname:<28} | {p_pct:>10.2f}% | {gt_str} | {gt_info}")
    print("=" * 85)

if __name__ == "__main__":
    evaluate_all()
