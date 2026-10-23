"""
Morphological mask refinement module.
Applies morphological operations to clean segmentation noise and fill small gaps,
while preserving thin linear structures (e.g., rivers, roads).
"""
import cv2
import numpy as np
from backend.app.core.config import settings

def refine_mask(class_mask_2d: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Refine multi-class segmentation mask using gentle morphological filtering.
    
    Args:
        class_mask_2d: (H, W) array of class IDs (1 to 5).
        kernel_size: Odd integer size for structuring element (default 3x3).
        
    Returns:
        Refined (H, W) class ID mask.
    """
    if class_mask_2d.ndim != 2:
        raise ValueError("class_mask_2d must be a 2D array.")
        
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    unique_classes = [c for c in np.unique(class_mask_2d) if c > 0]

    if len(unique_classes) <= 1:
        return class_mask_2d.copy()

    # Apply median filter to suppress salt-and-pepper pixel noise while preserving boundaries
    filtered = cv2.medianBlur(class_mask_2d.astype(np.uint8), kernel_size).astype(np.int32)

    # For each class, fill internal small holes using morphological close
    refined = filtered.copy()
    for c in unique_classes:
        bin_mask = (filtered == c).astype(np.uint8)
        if np.sum(bin_mask) == 0:
            continue
        closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        # Only adopt filled pixels if they were previously unassigned or ambiguous
        new_pixels = (closed == 1) & (filtered != c)
        # Apply gentle closing
        refined[new_pixels] = c

    return refined


def generate_colored_mask(class_mask_2d: np.ndarray) -> np.ndarray:
    """
    Generate an RGB visualization mask from a 2D class ID mask.
    """
    h, w = class_mask_2d.shape
    colored = np.full((h, w, 3), 128, dtype=np.uint8) # default grey

    id_to_color = {info["id"]: info["color_rgb"] for info in settings.CLASSES.values()}

    for class_id, color_rgb in id_to_color.items():
        colored[class_mask_2d == class_id] = color_rgb

    return colored
