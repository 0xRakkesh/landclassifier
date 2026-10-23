"""
Quantitative evaluation script for Phase E of the Land Classification roadmap.
Compares K-Means configurations (K=3, 5, 7) and morphological refinement on held-out Sentinel-2 images.
Outputs metrics table to outputs/evaluation_results.csv and creates comparison plots.
"""
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd

from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import assign_land_cover_labels
from backend.app.processing.refinement import refine_mask, generate_colored_mask
from backend.app.processing.metrics import compute_segmentation_metrics, calculate_area_statistics

def run_evaluation():
    base_dir = Path(__file__).resolve().parent.parent
    in_dir = base_dir / "data" / "input"
    gt_dir = base_dir / "data" / "ground_truth"
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_images = [
        "sentinel2_forest.jpg",
        "sentinel2_annualcrop.jpg",
        "sentinel2_industrial_builtup.jpg",
        "sentinel2_river_water.jpg",
        "sentinel2_sealake_water.jpg",
    ]

    k_values = [3, 5, 7]
    refinement_options = [False, True]

    records = []

    print("=" * 70)
    print("LAND CLASSIFICATION SYSTEM — QUANTITATIVE EVALUATION (PHASE E)")
    print("=" * 70)

    for img_name in test_images:
        img_path = in_dir / img_name
        gt_path = gt_dir / f"gt_{Path(img_name).stem}.png"

        if not img_path.exists() or not gt_path.exists():
            continue

        raw_bgr = load_image(img_path)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        # Preprocess to 256x256 to align with GT
        target_size = (gt_mask.shape[1], gt_mask.shape[0])
        resized_bgr, lab_image = preprocess_image(raw_bgr, target_size=target_size, apply_blur=True, blur_kernel=3)

        for k in k_values:
            for refine in refinement_options:
                t0 = time.perf_counter()

                # Segment
                cluster_labels, centers = segment_kmeans(lab_image, n_clusters=k, random_state=42)

                # Label
                class_mask, _, colored_rgb = assign_land_cover_labels(cluster_labels, centers)

                # Refine
                if refine:
                    eval_mask = refine_mask(class_mask, kernel_size=5)
                else:
                    eval_mask = class_mask

                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                # Compute supervised metrics against ground-truth
                metrics = compute_segmentation_metrics(eval_mask, gt_mask)

                records.append({
                    "image": img_name,
                    "k_clusters": k,
                    "morphological_refinement": refine,
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "mean_iou": metrics["mean_iou"],
                    "macro_f1": metrics["macro_f1"],
                    "processing_time_ms": elapsed_ms,
                })

                print(
                    f"[{img_name}] K={k:1d} | Refine={str(refine):<5} | "
                    f"Accuracy={metrics['pixel_accuracy']*100:5.1f}% | "
                    f"mIoU={metrics['mean_iou']:5.3f} | F1={metrics['macro_f1']:5.3f} | "
                    f"Time={elapsed_ms:5.1f}ms"
                )

    df_results = pd.DataFrame(records)
    csv_path = out_dir / "evaluation_results.csv"
    df_results.to_csv(csv_path, index=False)
    print("=" * 70)
    print(f"Evaluation metrics saved to: {csv_path}")

    # Summary table grouped by K and refinement
    summary = df_results.groupby(["k_clusters", "morphological_refinement"]).agg(
        avg_accuracy=("pixel_accuracy", "mean"),
        avg_mIoU=("mean_iou", "mean"),
        avg_f1=("macro_f1", "mean"),
        avg_time_ms=("processing_time_ms", "mean"),
    ).reset_index()

    summary_path = out_dir / "evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nBENCHMARK SUMMARY:")
    print(summary.to_string(index=False))

    return df_results

if __name__ == "__main__":
    run_evaluation()
