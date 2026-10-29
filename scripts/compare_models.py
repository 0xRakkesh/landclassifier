import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import assign_land_cover_labels
from backend.app.processing.random_forest import classify_random_forest
from backend.app.core.config import settings

img_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif"
mask_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\masks\M-34-51-C-b-2-1.tif"

img = load_image(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

resized_bgr, lab_image = preprocess_image(img, target_size=settings.DEFAULT_IMAGE_SIZE)
mask_resized = cv2.resize(mask, settings.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

# 1. Supervised Random Forest
t0 = time.time()
rf_mask, rf_rgb, rf_stats = classify_random_forest(resized_bgr)
t_rf = time.time() - t0

# 2. Unsupervised K-Means++
t0 = time.time()
km_labels, km_centers = segment_kmeans(lab_image, bgr_image=resized_bgr, n_clusters=5, random_state=42)
km_mask, km_map, km_rgb = assign_land_cover_labels(km_labels, km_centers, bgr_image=resized_bgr)
t_km = time.time() - t0

# Ground Truth Mapping:
# GT 0: Fields -> Pred 2
# GT 1: Building -> Pred 5
# GT 2: Woodland -> Pred 1 (Dense) or Pred 2 (Light)
# GT 3: Water -> Pred 3
# GT 4: Road -> Pred 4

def compute_metrics(pred_mask, gt_mask):
    total = gt_mask.size
    # Mapping ground truth to binary matches
    matches = 0
    ious = []
    
    # Class 3 (Water)
    gt_w = (gt_mask == 3)
    pred_w = (pred_mask == 3)
    if np.sum(gt_w) > 0:
        iou_w = np.sum(gt_w & pred_w) / (np.sum(gt_w | pred_w) + 1e-6)
        ious.append(iou_w)
    
    # Class 2 (Woodland)
    gt_f = (gt_mask == 2)
    pred_f = (pred_mask == 1) | (pred_mask == 2)
    if np.sum(gt_f) > 0:
        iou_f = np.sum(gt_f & pred_f) / (np.sum(gt_f | pred_f) + 1e-6)
        ious.append(iou_f)
        
    # Class 4 (Roads)
    gt_r = (gt_mask == 4)
    pred_r = (pred_mask == 4)
    if np.sum(gt_r) > 0:
        iou_r = np.sum(gt_r & pred_r) / (np.sum(gt_r | pred_r) + 1e-6)
        ious.append(iou_r)

    # Class 1 (Buildings)
    gt_b = (gt_mask == 1)
    pred_b = (pred_mask == 5)
    if np.sum(gt_b) > 0:
        iou_b = np.sum(gt_b & pred_b) / (np.sum(gt_b | pred_b) + 1e-6)
        ious.append(iou_b)

    mIoU = float(np.mean(ious)) if ious else 0.0
    return {
        "mIoU": round(mIoU * 100.0, 2),
        "water_iou": round(float(np.sum(gt_w & pred_w) / (np.sum(gt_w | pred_w) + 1e-6) * 100.0), 2),
        "forest_iou": round(float(np.sum(gt_f & pred_f) / (np.sum(gt_f | pred_f) + 1e-6) * 100.0), 2),
        "road_iou": round(float(np.sum(gt_r & pred_r) / (np.sum(gt_r | pred_r) + 1e-6) * 100.0), 2),
    }

metrics_rf = compute_metrics(rf_mask, mask_resized)
metrics_km = compute_metrics(km_mask, mask_resized)

print("--- MODEL COMPARISON ON M-34-51-C-b-2-1.tif ---")
print(f"Supervised Random Forest: mIoU = {metrics_rf['mIoU']}% | Time = {t_rf*1000:.1f}ms")
print(f"  Water IoU: {metrics_rf['water_iou']}% | Forest IoU: {metrics_rf['forest_iou']}% | Road IoU: {metrics_rf['road_iou']}%")
print(f"\nUnsupervised K-Means++:   mIoU = {metrics_km['mIoU']}% | Time = {t_km*1000:.1f}ms")
print(f"  Water IoU: {metrics_km['water_iou']}% | Forest IoU: {metrics_km['forest_iou']}% | Road IoU: {metrics_km['road_iou']}%")

best = "Supervised Random Forest (Calibrated on LandCover.ai)" if metrics_rf['mIoU'] >= metrics_km['mIoU'] else "Unsupervised K-Means++"
print(f"\nWINNER: {best}!")
