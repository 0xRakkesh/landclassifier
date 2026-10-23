"""
Land-cover area statistics and segmentation evaluation metrics.
Calculates class percentages, pixel accuracy, precision, recall, F1, and Intersection over Union (IoU).
"""
from typing import Dict, List, Any
import numpy as np
from backend.app.core.config import settings

def calculate_area_statistics(class_mask_2d: np.ndarray) -> List[Dict[str, Any]]:
    """
    Calculate pixel counts and percentage of the image covered by each land-cover category.
    
    Args:
        class_mask_2d: (H, W) array of class IDs (1 to 5).
        
    Returns:
        List of class statistics dictionaries.
    """
    total_pixels = class_mask_2d.size
    if total_pixels == 0:
        raise ValueError("Mask has zero pixels.")

    stats = []
    for class_name, info in settings.CLASSES.items():
        cid = info["id"]
        count = int(np.sum(class_mask_2d == cid))
        pct = round((count / total_pixels) * 100.0, 2)
        stats.append({
            "class_id": cid,
            "class_name": class_name,
            "color_hex": info["color_hex"],
            "color_rgb": info["color_rgb"],
            "pixel_count": count,
            "percentage": pct,
        })

    return stats


def compute_segmentation_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute quantitative evaluation metrics between predicted class mask and ground truth mask.
    
    Metrics:
        - Pixel Accuracy
        - Per-class Precision, Recall, F1-Score, and IoU
        - Mean IoU (mIoU) and Macro F1
    """
    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_mask.shape} vs ground truth {ground_truth_mask.shape}"
        )

    total_pixels = ground_truth_mask.size
    correct_pixels = int(np.sum(predicted_mask == ground_truth_mask))
    pixel_accuracy = round(float(correct_pixels / total_pixels), 4)

    per_class_metrics = {}
    ious = []
    f1s = []

    for class_name, info in settings.CLASSES.items():
        cid = info["id"]
        pred_pos = (predicted_mask == cid)
        gt_pos = (ground_truth_mask == cid)

        tp = int(np.sum(pred_pos & gt_pos))
        fp = int(np.sum(pred_pos & (~gt_pos)))
        fn = int(np.sum((~pred_pos) & gt_pos))

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        
        union = tp + fp + fn
        iou = round(tp / union, 4) if union > 0 else 0.0

        per_class_metrics[class_name] = {
            "class_id": cid,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "present_in_gt": bool(np.sum(gt_pos) > 0)
        }

        # Include in macro averages if class is present in ground truth
        if np.sum(gt_pos) > 0 or np.sum(pred_pos) > 0:
            ious.append(iou)
            f1s.append(f1)

    mean_iou = round(float(np.mean(ious)), 4) if ious else 0.0
    macro_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "total_evaluated_pixels": total_pixels,
    }
