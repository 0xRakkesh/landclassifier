import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.core.config import settings

img_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif"
img = load_image(img_path)
resized_bgr, lab_image = preprocess_image(
    img,
    target_size=settings.DEFAULT_IMAGE_SIZE,
    apply_blur=True,
    blur_kernel=5,
    use_bilateral=True,
)

# Extract texture map
gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
mean_int = cv2.blur(gray, (9, 9))
mean_sq = cv2.blur(gray * gray, (9, 9))
texture_map = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int))

cluster_labels_2d, cluster_centers_lab = segment_kmeans(
    lab_image,
    bgr_image=resized_bgr,
    n_clusters=5,
    random_state=42,
)

print("\n--- Cluster Detailed Profile with Texture ---")
for k in range(5):
    mask_k = (cluster_labels_2d == k)
    pct = np.sum(mask_k) / mask_k.size * 100
    b_mean = np.mean(resized_bgr[:, :, 0][mask_k])
    g_mean = np.mean(resized_bgr[:, :, 1][mask_k])
    r_mean = np.mean(resized_bgr[:, :, 2][mask_k])
    
    L_mean = np.mean(lab_image[:, :, 0][mask_k])
    a_mean = np.mean(lab_image[:, :, 1][mask_k])
    lab_b_mean = np.mean(lab_image[:, :, 2][mask_k])
    
    tex_mean = np.mean(texture_map[mask_k])
    exg = 2.0 * g_mean - r_mean - b_mean
    wai = (b_mean - r_mean) / (b_mean + r_mean + 1e-4)

    hsv = cv2.cvtColor(np.uint8([[[b_mean, g_mean, r_mean]]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue, sat = hsv[0], hsv[1]

    print(f"Cluster {k} ({pct:.1f}%):")
    print(f"  RGB=({r_mean:.1f}, {g_mean:.1f}, {b_mean:.1f}), LAB=({L_mean:.1f}, {a_mean:.1f}, {lab_b_mean:.1f})")
    print(f"  Texture Std = {tex_mean:.2f} (Roughness)")
    print(f"  ExG = {exg:.1f}, WAI = {wai:.3f}, Sat = {sat}")
