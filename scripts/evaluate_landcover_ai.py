"""
Evaluation script to benchmark the pipeline against LandCover.ai Ground Truth masks.

LandCover.ai Ground Truth mapping:
  0: Other (Agricultural fields, meadows, bare soil, grassland)
  1: Building
  2: Woodland (Dense Forest / Trees)
  3: Water
  4: Road

Pipeline Classes:
  1: Dense Forest -> matches GT 2 (Woodland)
  2: Farm Fields / Light Forest -> matches GT 0 (Fields / Other vegetation)
  3: Water Bodies -> matches GT 3 (Water)
  4: Roads / Highways -> matches GT 4 (Road)
  5: Buildings / Dry Land -> matches GT 1 (Building) + dry portions of GT 0
"""

import os
import sys
import glob
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from backend.app.processing.pipeline import classify_satellite_image

DATASET_DIR = r"c:\Users\rakes\OneDrive\Desktop\landcover.ai.v1"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")

def evaluate_sample(image_path: str, mask_path: str, crop_size: int = 1500):
    print(f"\nEvaluating: {os.path.basename(image_path)}")
    
    # Read image and mask
    img_bgr = cv2.imread(image_path)
    mask_gt = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    if img_bgr is None or mask_gt is None:
        print("Error loading image or mask.")
        return None

    # Take a representative central crop for fast detailed evaluation
    h, w = img_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    
    crop_img_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    crop_gt = mask_gt[y1:y2, x1:x2]

    # Encode crop to bytes to pass into our pipeline
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(crop_img_rgb, cv2.COLOR_RGB2BGR))
    if not is_success:
        print("Failed to encode crop")
        return None
    
    t0 = time.time()
    result = classify_satellite_image(buffer.tobytes(), n_clusters=5, max_dim=crop_size)
    elapsed = time.time() - t0
    
    pred_2d = result["class_mask_2d"]
    # Resize GT if pipeline resized the image
    if crop_gt.shape != pred_2d.shape:
        crop_gt = cv2.resize(crop_gt, (pred_2d.shape[1], pred_2d.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    print(f"Crop shape: {pred_2d.shape}, Pipeline time: {elapsed:.2f}s")
    
    # Classes in GT
    gt_unique, gt_counts = np.unique(crop_gt, return_counts=True)
    print(f"Ground Truth distribution: {dict(zip(gt_unique.tolist(), (gt_counts/crop_gt.size*100).round(1).tolist()))}%")
    
    # Pipeline predicted distribution
    pred_unique, pred_counts = np.unique(pred_2d, return_counts=True)
    print(f"Predicted distribution: {dict(zip(pred_unique.tolist(), (pred_counts/pred_2d.size*100).round(1).tolist()))}%")
    
    # Evaluate specific key classes:
    # 1. Water (GT 3 vs Pred 3)
    if 3 in gt_unique:
        gt_water = (crop_gt == 3)
        pred_water = (pred_2d == 3)
        intersection = np.sum(gt_water & pred_water)
        union = np.sum(gt_water | pred_water)
        iou_water = (intersection / union) if union > 0 else 0
        acc_water = np.sum(pred_water == gt_water) / crop_gt.size
        print(f"  [Water] GT pixels: {np.sum(gt_water)}, IoU: {iou_water:.3f}, TP: {intersection}, FP: {np.sum(pred_water & ~gt_water)}")

    # 2. Woodland / Dense Forest (GT 2 vs Pred 1)
    if 2 in gt_unique:
        gt_forest = (crop_gt == 2)
        # In our model, vegetation can be Dense Forest (1) or Light Forest / Fields (2)
        pred_forest = (pred_2d == 1) | (pred_2d == 2)
        intersection = np.sum(gt_forest & pred_forest)
        union = np.sum(gt_forest | pred_forest)
        iou_forest = (intersection / union) if union > 0 else 0
        print(f"  [Forest / Vegetation] GT Woodland pixels: {np.sum(gt_forest)}, Overlap with Forest/Fields: {intersection / np.sum(gt_forest) * 100:.1f}%, IoU: {iou_forest:.3f}")

    # 3. Roads (GT 4 vs Pred 4)
    if 4 in gt_unique:
        gt_road = (crop_gt == 4)
        pred_road = (pred_2d == 4)
        intersection = np.sum(gt_road & pred_road)
        union = np.sum(gt_road | pred_road)
        iou_road = (intersection / union) if union > 0 else 0
        recall_road = intersection / np.sum(gt_road) if np.sum(gt_road) > 0 else 0
        print(f"  [Roads] GT Road pixels: {np.sum(gt_road)}, Recall: {recall_road*100:.1f}%, IoU: {iou_road:.3f}")

    # 4. Buildings (GT 1 vs Pred 5)
    if 1 in gt_unique:
        gt_bldg = (crop_gt == 1)
        pred_bldg = (pred_2d == 5)
        intersection = np.sum(gt_bldg & pred_bldg)
        recall_bldg = intersection / np.sum(gt_bldg) if np.sum(gt_bldg) > 0 else 0
        print(f"  [Buildings] GT Building pixels: {np.sum(gt_bldg)}, Recall: {recall_bldg*100:.1f}%")

if __name__ == "__main__":
    imgs = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.tif")))
    masks = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))
    
    print(f"Found {len(imgs)} images and {len(masks)} masks.")
    # Test across 3 diverse tiles:
    # 1 containing water/river
    # 1 containing urban/roads/buildings
    # 1 containing rural/forest
    for i in [0, 5, 10]:
        evaluate_sample(imgs[i], masks[i])
