import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.services.classifier import LandClassifierService
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.core.config import settings

img_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif"
img = load_image(img_path)
resized_bgr, lab_image = preprocess_image(img, target_size=settings.DEFAULT_IMAGE_SIZE)
gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
ksize = 9
mean_int = cv2.blur(gray, (ksize, ksize))
mean_sq = cv2.blur(gray * gray, (ksize, ksize))
texture_map = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int))

labels, centers = segment_kmeans(lab_image, bgr_image=resized_bgr, n_clusters=5, random_state=42)

for k in range(5):
    mask = (labels == k)
    pct = np.sum(mask) / labels.size * 100
    b_m = np.mean(resized_bgr[:, :, 0][mask])
    g_m = np.mean(resized_bgr[:, :, 1][mask])
    r_m = np.mean(resized_bgr[:, :, 2][mask])
    L_m = np.mean(lab_image[:, :, 0][mask])
    a_m = np.mean(lab_image[:, :, 1][mask])
    b_lab_m = np.mean(lab_image[:, :, 2][mask])
    tex_m = np.mean(texture_map[mask])
    exg = 2.0 * g_m - r_m - b_m
    wai = (b_m - r_m) / (b_m + r_m + 1e-4)
    hsv = cv2.cvtColor(np.uint8([[[b_m, g_m, r_m]]]), cv2.COLOR_BGR2HSV)[0, 0]
    print(f"Cluster {k} ({pct:.1f}%):")
    print(f"  RGB=({r_m:.1f}, {g_m:.1f}, {b_m:.1f}), LAB=({L_m:.1f}, {a_m:.1f}, {b_lab_m:.1f})")
    print(f"  Texture Std={tex_m:.2f}, ExG={exg:.1f}, WAI={wai:.3f}, Sat={hsv[1]}, Hue={hsv[0]}")
