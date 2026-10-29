"""
LandClassifierService: High-level facade for end-to-end land cover classification pipeline.
Primary Engine: Supervised Random Forest (Calibrated on LandCover.ai Ground Truth).
Provides live accuracy, precision, recall, and IoU evaluation whenever ground truth is present.
Also supports K-Means clustering for backward compatibility.
"""
import base64
import time
import os
import json
from typing import Dict, Any, Optional
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from backend.app.core.config import settings
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import assign_land_cover_labels
from backend.app.processing.refinement import refine_mask, generate_colored_mask
from backend.app.processing.metrics import calculate_area_statistics
from backend.app.processing.random_forest import classify_random_forest, GT_TO_APP_ID

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_metrics.json")

class LandClassifierService:
    @staticmethod
    def _to_base64_png(image_arr: np.ndarray, is_rgb: bool = False) -> str:
        """Helper to encode image to base64 PNG string."""
        if is_rgb:
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", image_arr)
        if not success:
            raise RuntimeError("Failed to encode image to PNG.")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @classmethod
    def get_benchmark_metrics(cls) -> Dict[str, Any]:
        """Load offline test-set benchmark metrics."""
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @classmethod
    def classify(
        cls,
        image_bytes: bytes,
        file_name: Optional[str] = None,
        algorithm: str = "random_forest",
        n_clusters: int = 5,
        apply_blur: bool = True,
        blur_kernel: int = 5,
        use_bilateral: bool = True,
        refine: bool = True,
        refine_kernel: int = 3,
        gt_mask_bytes: Optional[bytes] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute land cover classification using Supervised Random Forest (or K-Means if requested).
        Computes live accuracy, precision, recall, and IoU if ground truth mask is present or auto-detected.
        """
        start_time = time.perf_counter()

        # 1. Load and validate
        raw_bgr = load_image(image_bytes, max_dimension=settings.MAX_IMAGE_DIMENSION)
        orig_h, orig_w = raw_bgr.shape[:2]

        # 2. Preprocess with edge-preserving bilateral filter
        resized_bgr, lab_image = preprocess_image(
            raw_bgr,
            target_size=settings.DEFAULT_IMAGE_SIZE,
            apply_blur=apply_blur,
            blur_kernel=blur_kernel,
            use_bilateral=use_bilateral,
        )

        cluster_mapping = {}

        # 3. Model Classification
        if algorithm.lower() == "kmeans":
            cluster_labels_2d, cluster_centers_lab = segment_kmeans(
                lab_image,
                bgr_image=resized_bgr,
                n_clusters=n_clusters,
                random_state=42,
            )
            class_mask_2d, cluster_to_class, initial_colored_rgb = assign_land_cover_labels(
                cluster_labels_2d,
                cluster_centers_lab,
                bgr_image=resized_bgr,
            )
            if refine:
                final_class_mask = refine_mask(class_mask_2d, kernel_size=refine_kernel)
                final_colored_rgb = generate_colored_mask(final_class_mask)
            else:
                final_class_mask = class_mask_2d
                final_colored_rgb = initial_colored_rgb

            area_stats = calculate_area_statistics(final_class_mask)
            cluster_mapping = {f"cluster_{k}": cls_name for k, cls_name in cluster_to_class.items()}
            algo_name = "kmeans"
        else:
            final_class_mask, final_colored_rgb, area_stats = classify_random_forest(
                resized_bgr,
                refine=refine,
                refine_kernel=refine_kernel,
            )
            algo_name = "Supervised Random Forest (LandCover.ai)"

        # 4. Live Ground Truth Evaluation (if mask provided or matching file found in data/masks)
        gt_arr = None
        if gt_mask_bytes is not None and len(gt_mask_bytes) > 0:
            gt_arr = cv2.imdecode(np.frombuffer(gt_mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        elif file_name:
            basename = os.path.basename(file_name)
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "masks", basename),
                os.path.join(r"C:\Users\rakes\OneDrive\Desktop\landclassifier\data\masks", basename),
                os.path.join(r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\masks", basename),
            ]
            for c_path in candidates:
                if os.path.exists(c_path):
                    gt_arr = cv2.imread(c_path, cv2.IMREAD_UNCHANGED)
                    break

        evaluation = {"has_ground_truth": False}
        if gt_arr is not None:
            gt_resized = cv2.resize(gt_arr, settings.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            if gt_resized.ndim == 3:
                gt_resized = gt_resized[:, :, 0]

            gt_app = np.zeros_like(gt_resized, dtype=np.int32)
            for g_id, a_id in GT_TO_APP_ID.items():
                gt_app[gt_resized == g_id] = a_id

            y_true = gt_app.reshape(-1)
            y_pred = final_class_mask.reshape(-1)

            acc = float(accuracy_score(y_true, y_pred) * 100.0)
            active_labels = [1, 2, 3, 5] if 4 not in GT_TO_APP_ID.values() else [1, 2, 3, 4, 5]
            prec, rec, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=active_labels, zero_division=0
            )

            per_class_info = {}
            valid_ious = []
            class_map = {
                1: "Dense Forest",
                2: "Farm Fields / Light Forest",
                3: "Water Bodies",
                4: "Roads / Highways",
                5: "Buildings, Roads & Dry Land",
            }
            for idx, cid in enumerate(active_labels):
                t_b = (y_true == cid)
                p_b = (y_pred == cid)
                inter = np.sum(t_b & p_b)
                union = np.sum(t_b | p_b)
                iou_val = float(inter / union * 100.0) if union > 0 else 0.0
                if np.sum(t_b) > 0:
                    valid_ious.append(iou_val)

                cname = class_map[cid]
                per_class_info[cname] = {
                    "precision": round(float(prec[idx] * 100.0), 2),
                    "recall": round(float(rec[idx] * 100.0), 2),
                    "iou": round(iou_val, 2),
                    "f1": round(float(f1[idx] * 100.0), 2),
                    "support": int(support[idx]),
                }

            macro_p = round(float(np.mean(prec) * 100.0), 2)
            macro_r = round(float(np.mean(rec) * 100.0), 2)
            macro_f = round(float(np.mean(f1) * 100.0), 2)
            miou = round(float(np.mean(valid_ious)), 2) if valid_ious else 0.0

            evaluation = {
                "has_ground_truth": True,
                "accuracy": round(acc, 2),
                "precision": macro_p,
                "recall": macro_r,
                "f1_score": macro_f,
                "mean_iou": miou,
                "per_class": per_class_info,
            }

        orig_b64 = cls._to_base64_png(resized_bgr, is_rgb=False)
        segmented_b64 = cls._to_base64_png(final_colored_rgb, is_rgb=True)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return {
            "status": "success",
            "algorithm": algo_name,
            "algorithm_key": algorithm,
            "processing_time_ms": elapsed_ms,
            "original_dimensions": {"width": orig_w, "height": orig_h},
            "processed_dimensions": {
                "width": settings.DEFAULT_IMAGE_SIZE[0],
                "height": settings.DEFAULT_IMAGE_SIZE[1],
            },
            "parameters": {
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "apply_blur": apply_blur,
                "blur_kernel": blur_kernel,
                "use_bilateral": use_bilateral,
                "refine_mask": refine,
                "refine_kernel": refine_kernel,
            },
            "cluster_mapping": cluster_mapping,
            "classes": area_stats,
            "original_image_base64": orig_b64,
            "segmented_image_base64": segmented_b64,
            "evaluation": evaluation,
        }
