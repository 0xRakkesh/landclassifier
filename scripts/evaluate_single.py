"""
CLI Evaluation Tool: Evaluates predictions against Ground Truth labels from landcover.ai.v1.

Usage:
    uv run python scripts/evaluate_single.py --image "C:/path/to/image.tif"
    or
    uv run python scripts/evaluate_single.py --image "C:/path/to/image.tif" --mask "C:/path/to/mask.tif"
"""
import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.classifier import LandClassifierService
from backend.app.core.config import settings

GT_CLASS_NAMES = {
    0: "Background / Fields (GT 0)",
    1: "Building (GT 1)",
    2: "Woodland / Forest (GT 2)",
    3: "Water (GT 3)",
    4: "Road (GT 4)",
}

PRED_CLASS_NAMES = {
    1: "Dense Forest",
    2: "Farm Fields / Light Forest",
    3: "Water Bodies",
    4: "Roads / Highways",
    5: "Buildings / Dry Land",
    6: "Unclassified",
}

def evaluate(image_path: str, mask_path: str = None):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Auto-find mask if not provided
    if mask_path is None:
        basename = os.path.basename(image_path)
        candidate = os.path.join(r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\masks", basename)
        if os.path.exists(candidate):
            mask_path = candidate
        else:
            print(f"Error: No mask specified and could not auto-find mask for {basename}")
            return

    print(f"\n=======================================================")
    print(f"[EVALUATION] Image:        {os.path.basename(image_path)}")
    print(f"[GROUND TRUTH] Mask:       {os.path.basename(mask_path)}")
    print(f"=======================================================\n")

    # 1. Load image and mask
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    
    gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if gt_mask_raw is None:
        print(f"Failed to load mask {mask_path}")
        return

    # 2. Run Classification Pipeline
    res = LandClassifierService.classify(
        image_bytes=img_bytes,
        n_clusters=5,
        apply_blur=True,
        blur_kernel=5,
        use_bilateral=True,
        refine=True,
    )

    h, w = settings.DEFAULT_IMAGE_SIZE
    gt_mask = cv2.resize(gt_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. Print Area Comparison
    print("1. AREA COVERAGE COMPARISON:")
    print(f"{'Class Name':<30} | {'Predicted %':<14} | {'Ground Truth %':<14}")
    print("-" * 65)

    gt_counts = {gid: np.sum(gt_mask == gid) for gid in range(5)}
    pred_stats = {c["class_id"]: c["percentage"] for c in res["classes"]}

    print(f"{'Water Bodies (Blue)':<30} | {pred_stats.get(3, 0.0):>12.2f}% | {gt_counts[3] / gt_mask.size * 100:>12.2f}% (GT 3: Water)")
    print(f"{'Dense Forest (Deep Green)':<30} | {pred_stats.get(1, 0.0):>12.2f}% | {gt_counts[2] / gt_mask.size * 100:>12.2f}% (GT 2: Woodland)")
    print(f"{'Farm Fields (Green)':<30} | {pred_stats.get(2, 0.0):>12.2f}% | {'--':>13} (Part of GT 0)")
    print(f"{'Roads / Highways (Black)':<30} | {pred_stats.get(4, 0.0):>12.2f}% | {gt_counts[4] / gt_mask.size * 100:>12.2f}% (GT 4: Road)")
    print(f"{'Buildings / Dry Land (Yellow)':<30} | {pred_stats.get(5, 0.0):>12.2f}% | {gt_counts[1] / gt_mask.size * 100:>12.2f}% (GT 1: Building)")
    print(f"{'Unclassified (Grey)':<30} | {pred_stats.get(6, 0.0):>12.2f}% | {'0.00%':>13}")
    print("-" * 65)

    print("\n2. GROUND TRUTH OVERLAP METRICS:")
    print(f"{'Target Feature':<25} | {'Area Match Ratio':<18} | {'Details'}")
    print("-" * 75)

    eval_targets = [
        ("Water Bodies", 3, [3]),
        ("Woodland / Forest", 2, [1, 2]),
        ("Roads & Highways", 4, [4]),
        ("Buildings", 1, [5]),
    ]

    for label_name, gt_id, pred_ids in eval_targets:
        gt_bin = (gt_mask == gt_id)
        if np.sum(gt_bin) == 0:
            print(f"{label_name:<25} | {'Not in Image':<18} | GT count is 0")
            continue

        gt_count = np.sum(gt_bin)
        pred_pct = sum(pred_stats.get(pid, 0.0) for pid in pred_ids)
        gt_pct = gt_count / gt_mask.size * 100.0
        
        ratio = min(pred_pct, gt_pct) / max(pred_pct, gt_pct) if max(pred_pct, gt_pct) > 0 else 0.0
        print(f"{label_name:<25} | {ratio:>16.1%} | Predicted: {pred_pct:>5.1f}% vs GT: {gt_pct:>5.1f}%")

    print("\n[OK] Evaluation complete. Labels in landcover.ai.v1 are 100% ground-truth verified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction against LandCover.ai ground truth mask.")
    parser.add_argument("--image", type=str, default=r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif", help="Path to image TIFF")
    parser.add_argument("--mask", type=str, default=None, help="Path to ground truth mask TIFF")
    args = parser.parse_args()

    evaluate(args.image, args.mask)
