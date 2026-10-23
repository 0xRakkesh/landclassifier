"""
Image loading and preprocessing module.
Handles validation, resizing, noise reduction (Gaussian blur), and CIELAB color space conversion.
"""
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np

def load_image(image_input: Union[bytes, str, Path], max_dimension: int = 4096) -> np.ndarray:
    """
    Load an image from bytes, file path, or Path object into BGR format.
    Validates dimensions and ensures 3-channel format.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        image_bytes = path.read_bytes()
    elif isinstance(image_input, bytes):
        image_bytes = image_input
    else:
        raise TypeError(f"Expected bytes, str, or Path, got {type(image_input)}")

    if len(image_bytes) == 0:
        raise ValueError("Image input buffer is empty.")

    # Decode from memory
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Failed to decode image. Format may be unsupported or file is corrupted.")

    # Convert grayscale or 4-channel BGRA to 3-channel BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Image dimensions must be non-zero.")
    if h > max_dimension or w > max_dimension:
        raise ValueError(f"Image dimension ({w}x{h}) exceeds allowed maximum of {max_dimension}px.")

    return image


def preprocess_image(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    apply_blur: bool = True,
    blur_kernel: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image: resize, optional Gaussian blur, and CIELAB conversion.
    
    Args:
        image_bgr: Input BGR image array.
        target_size: (width, height) to resize.
        apply_blur: Whether to apply Gaussian blur to reduce noise.
        blur_kernel: Kernel size for Gaussian blur (must be positive odd integer).

    Returns:
        Tuple of (resized_bgr_image, lab_image).
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel numpy array (BGR).")

    # Resize image
    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Target size dimensions must be positive integers, got {target_size}")

    h, w = image_bgr.shape[:2]
    interpolation = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
    resized_bgr = cv2.resize(image_bgr, (target_w, target_h), interpolation=interpolation)

    # Optional Gaussian Blur for noise reduction
    working_bgr = resized_bgr.copy()
    if apply_blur:
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be a positive odd integer, got {blur_kernel}")
        working_bgr = cv2.GaussianBlur(working_bgr, (blur_kernel, blur_kernel), 0)

    # Convert BGR to CIELAB (L*a*b*)
    lab_image = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)

    return resized_bgr, lab_image
