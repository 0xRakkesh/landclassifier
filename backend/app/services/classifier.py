"""
LandClassifierService: High-level facade for end-to-end land cover classification pipeline.
"""
import base64
import time
from typing import Dict, Any
import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import assign_land_cover_labels
from backend.app.processing.refinement import refine_mask, generate_colored_mask
from backend.app.processing.metrics import calculate_area_statistics

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
    def classify(
        cls,
        image_bytes: bytes,
        n_clusters: int = 5,
        apply_blur: bool = True,
        blur_kernel: int = 3,
        refine: bool = True,
        refine_kernel: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end land cover classification pipeline.
        
        Args:
            image_bytes: Raw uploaded image bytes.
            n_clusters: Number of clusters (3, 5, or 7).
            apply_blur: Toggle Gaussian blur.
            blur_kernel: Odd integer blur kernel size.
            refine: Toggle morphological mask refinement.
            refine_kernel: Odd integer morphological kernel size.
            
        Returns:
            Dictionary containing structured statistics, timings, and base64 images.
        """
        start_time = time.perf_counter()

        # 1. Load and validate
        raw_bgr = load_image(image_bytes, max_dimension=settings.MAX_IMAGE_DIMENSION)
        orig_h, orig_w = raw_bgr.shape[:2]

        # 2. Preprocess
        resized_bgr, lab_image = preprocess_image(
            raw_bgr,
            target_size=settings.DEFAULT_IMAGE_SIZE,
            apply_blur=apply_blur,
            blur_kernel=blur_kernel,
        )

        # 3. K-Means segmentation in CIELAB
        cluster_labels_2d, cluster_centers_lab = segment_kmeans(
            lab_image,
            n_clusters=n_clusters,
            random_state=42,
        )

        # 4. Semantic labelling
        class_mask_2d, cluster_to_class, initial_colored_rgb = assign_land_cover_labels(
            cluster_labels_2d,
            cluster_centers_lab,
        )

        # 5. Mask refinement
        if refine:
            final_class_mask = refine_mask(class_mask_2d, kernel_size=refine_kernel)
            final_colored_rgb = generate_colored_mask(final_class_mask)
        else:
            final_class_mask = class_mask_2d
            final_colored_rgb = initial_colored_rgb

        # 6. Area statistics
        area_stats = calculate_area_statistics(final_class_mask)

        # 7. Base64 encoding
        orig_b64 = cls._to_base64_png(resized_bgr, is_rgb=False)
        segmented_b64 = cls._to_base64_png(final_colored_rgb, is_rgb=True)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return {
            "status": "success",
            "processing_time_ms": elapsed_ms,
            "original_dimensions": {"width": orig_w, "height": orig_h},
            "processed_dimensions": {
                "width": settings.DEFAULT_IMAGE_SIZE[0],
                "height": settings.DEFAULT_IMAGE_SIZE[1],
            },
            "parameters": {
                "n_clusters": n_clusters,
                "apply_blur": apply_blur,
                "blur_kernel": blur_kernel,
                "refine_mask": refine,
                "refine_kernel": refine_kernel,
            },
            "cluster_mapping": {
                f"cluster_{k}": cls_name for k, cls_name in cluster_to_class.items()
            },
            "classes": area_stats,
            "original_image_base64": orig_b64,
            "segmented_image_base64": segmented_b64,
        }
