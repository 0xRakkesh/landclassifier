import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import analyze_centroid
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

cluster_labels_2d, cluster_centers_lab = segment_kmeans(
    lab_image,
    bgr_image=resized_bgr,
    n_clusters=5,
    random_state=42,
)

print("\n--- Centroid Detailed Features ---")
for k in range(5):
    c = analyze_centroid(cluster_centers_lab[k])
    count = np.sum(cluster_labels_2d == k)
    pct = count / cluster_labels_2d.size * 100
    print(f"Cluster {k} ({pct:.1f}% of image):")
    print(f"  L={c['L']:.1f}, a={c['a']:.1f}, b={c['b']:.1f}")
    print(f"  RGB=({c['r']:.1f}, {c['g']:.1f}, {c['blue']:.1f})")
    print(f"  Hue={c['hue']:.1f}, Sat={c['sat']:.1f}, Val={c['val']:.1f}")
    print(f"  ExG={c['exg']:.1f}, WAI={c['wai']:.3f}")
