import sys
import os
import glob
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.classifier import LandClassifierService
from backend.app.core.config import settings

DATASET_DIR = r"c:\Users\rakes\OneDrive\Desktop\landcover.ai.v1"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")

def main():
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))

    print(f"Starting evaluation across {len(img_paths)} images...")
    
    # Aggregated pixel counts
    # GT classes: 0: Other, 1: Building, 2: Woodland, 3: Water, 4: Road
    # Pred classes: 1: Dense Forest, 2: Farm Fields, 3: Water, 4: Roads, 5: Buildings/Dry, 6: Unclassified
    
    gt_total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    pred_total_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    # Confusion matrix: GT (0..4) x Pred (1..6)
    # Rows: GT, Cols: Pred
    conf_matrix = np.zeros((5, 7), dtype=np.int64)

    image_results = []
    t_start = time.time()

    for idx, (img_p, mask_p) in enumerate(zip(img_paths, mask_paths)):
        name = os.path.splitext(os.path.basename(img_p))[0]
        
        # Load mask
        mask_gt = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        if mask_gt is None:
            print(f"Failed to load mask {mask_p}")
            continue

        # Load image bytes and classify
        with open(img_p, "rb") as f:
            img_bytes = f.read()

        t0 = time.time()
        res = LandClassifierService.classify(
            image_bytes=img_bytes,
            n_clusters=5,
            apply_blur=True,
            blur_kernel=5,
            use_bilateral=True,
            refine=True,
        )
        elapsed = time.time() - t0

        # Resize GT mask to match processed dimensions (512, 512)
        h, w = settings.DEFAULT_IMAGE_SIZE
        mask_gt_resized = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)

        # Extract predicted class mask
        # We can reconstruct it from classes or call assign_land_cover_labels directly
        # Let's decode segmented base64 or calculate directly
        # Or faster: run classification pipeline returning mask
        classes_stats = {c["class_id"]: c["percentage"] for c in res["classes"]}

        # Count GT pixels in resized
        for gid in range(5):
            count = int(np.sum(mask_gt_resized == gid))
            gt_total_counts[gid] += count

        # Count Pred pixels
        for pid in range(1, 7):
            pct = classes_stats.get(pid, 0.0)
            count = int(np.round(pct / 100.0 * (h * w)))
            pred_total_counts[pid] += count

        # Per image GT percentages
        gt_pcts = {
            gid: round(float(np.sum(mask_gt_resized == gid) / (h * w) * 100.0), 2)
            for gid in range(5)
        }

        image_results.append({
            "name": name,
            "time_s": round(elapsed, 2),
            "gt_water": gt_pcts[3],
            "pred_water": classes_stats.get(3, 0.0),
            "gt_woodland": gt_pcts[2],
            "pred_dense_forest": classes_stats.get(1, 0.0),
            "pred_farm_fields": classes_stats.get(2, 0.0),
            "gt_road": gt_pcts[4],
            "pred_road": classes_stats.get(4, 0.0),
            "gt_building": gt_pcts[1],
            "pred_building_dry": classes_stats.get(5, 0.0),
            "gt_other": gt_pcts[0],
            "pred_unclassified": classes_stats.get(6, 0.0),
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(img_paths):
            print(f"[{idx + 1}/{len(img_paths)}] Processed {name} in {elapsed:.2f}s")

    total_time = time.time() - t_start
    print(f"\nAll {len(img_paths)} images processed in {total_time:.1f}s!")

    # Dataset Total Percentages
    total_gt_pixels = sum(gt_total_counts.values())
    total_pred_pixels = sum(pred_total_counts.values())

    gt_agg_pct = {k: round(v / total_gt_pixels * 100.0, 2) for k, v in gt_total_counts.items()}
    pred_agg_pct = {k: round(v / total_pred_pixels * 100.0, 2) for k, v in pred_total_counts.items()}

    print("\n================ DATASET-WIDE MATRIX ================")
    print(f"{'Land-Cover Class':<28} | {'Pred Area %':<12} | {'LandCover.ai GT %':<18} | {'GT Class Name'}")
    print("-" * 80)
    print(f"{'Water Bodies':<28} | {pred_agg_pct[3]:>10}% | {gt_agg_pct[3]:>16}% | Water (GT 3)")
    print(f"{'Dense Forest':<28} | {pred_agg_pct[1]:>10}% | {gt_agg_pct[2]:>16}% | Woodland / Trees (GT 2)")
    print(f"{'Farm Fields / Light Forest':<28} | {pred_agg_pct[2]:>10}% | {'--':>17} | Cropland / Grass (GT 0)")
    print(f"{'Roads / Highways':<28} | {pred_agg_pct[4]:>10}% | {gt_agg_pct[4]:>16}% | Road network (GT 4)")
    print(f"{'Buildings / Dry Land':<28} | {pred_agg_pct[5]:>10}% | {gt_agg_pct[1]:>16}% | Buildings (GT 1) + Soil")
    print(f"{'Unclassified':<28} | {pred_agg_pct[6]:>10}% | {'0.00%':>17} | Shadows / Mixed")
    print("=" * 80)

    # Save detailed per-image results to CSV
    import csv
    csv_file = "evaluation_matrix_all_images.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_results[0].keys()))
        writer.writeheader()
        writer.writerows(image_results)
    print(f"\nSaved per-image detailed evaluation to {csv_file}")

if __name__ == "__main__":
    main()
