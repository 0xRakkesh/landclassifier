"""
Image loading and preprocessing module.
Handles validation, resizing, robust dynamic range normalization (DIP Lab Program 5),
edge-preserving bilateral filtering or Gaussian blur, and CIELAB color space conversion.
"""
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np

def load_image(image_input: Union[bytes, str, Path], max_dimension: int = 16384) -> np.ndarray:
    """
    Load an image from bytes, file path, or Path object into BGR format.
    Supports large multi-MB satellite TIFF, PNG, and JPEG imagery.
    Validates dimensions and ensures 3-channel 8-bit format.
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

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    # Fallback to PIL for complex GeoTIFF or unsupported TIFF encodings
    if image is None:
        try:
            from PIL import Image as PILImage
            import io
            PILImage.MAX_IMAGE_PIXELS = None
            pil_img = PILImage.open(io.BytesIO(image_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as err:
            raise ValueError(f"Failed to decode image: {err}")

    if image is None:
        raise ValueError("Failed to decode image. Format may be unsupported or file is corrupted.")

    # Convert 16-bit or float TIFFs to 8-bit standard
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

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


def normalize_dynamic_range(image_bgr: np.ndarray) -> np.ndarray:
    """
    Robust percentile-based contrast stretching (DIP Lab Program 5).
    Stretches 1st to 99th percentile across [0, 255].
    Eliminates washed-out display haze and monitor gamma shifts in screenshots,
    restoring true ground-surface spectral dynamic range.
    """
    normalized = np.empty_like(image_bgr)
    for c in range(3):
        channel = image_bgr[:, :, c]
        p_low, p_high = np.percentile(channel, (1.0, 99.0))
        if p_high > p_low + 10:
            stretched = np.clip((channel.astype(float) - p_low) / (p_high - p_low) * 255.0, 0, 255)
            normalized[:, :, c] = stretched.astype(np.uint8)
        else:
            normalized[:, :, c] = channel
    return normalized


def preprocess_image(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    apply_blur: bool = True,
    blur_kernel: int = 5,
    use_bilateral: bool = True,
    normalize_contrast: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image: resize, dynamic range normalization, edge-preserving filtering, and CIELAB conversion.
    
    Args:
        image_bgr: Input BGR image array.
        target_size: (width, height) to resize.
        apply_blur: Whether to apply smoothing filter to reduce sensor noise.
        blur_kernel: Kernel diameter / size (positive odd integer).
        use_bilateral: If True, uses Bilateral Filter to smooth noise while preserving sharp edges.
        normalize_contrast: If True, applies robust percentile contrast normalization for screenshots.

    Returns:
        Tuple of (resized_bgr_image, lab_image).
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel numpy array (BGR).")

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Target size dimensions must be positive integers, got {target_size}")

    h, w = image_bgr.shape[:2]
    interpolation = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
    resized_bgr = cv2.resize(image_bgr, (target_w, target_h), interpolation=interpolation)

    # 1. Normalize dynamic range to fix washed-out screenshots / display capture gamma shifts
    if normalize_contrast:
        working_bgr = normalize_dynamic_range(resized_bgr)
    else:
        working_bgr = resized_bgr.copy()

    # 2. Edge-preserving smoothing filter
    if apply_blur:
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be a positive odd integer, got {blur_kernel}")
        if use_bilateral:
            working_bgr = cv2.bilateralFilter(working_bgr, d=blur_kernel, sigmaColor=75, sigmaSpace=75)
        else:
            working_bgr = cv2.GaussianBlur(working_bgr, (blur_kernel, blur_kernel), 0)

    # 3. Convert to CIELAB perceptual color space
    lab_image = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)

    return working_bgr, lab_image
