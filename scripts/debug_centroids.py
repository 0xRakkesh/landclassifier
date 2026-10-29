import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import analyze_centroid, assign_land_cover_labels

img_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif"
img = load_image(img_path)
bgr, lab = preprocess_image(img, target_size=(1024, 1024))
labels, centers = segment_kmeans(lab, n_clusters=5)

print("\n--- Centroid Analysis ---")
for i, c in enumerate(centers):
    info = analyze_centroid(c)
    print(f"Cluster {i}: L={info['L']:.1f}, a={info['a']:.1f}, b={info['b']:.1f}, "
          f"RGB=({info['r']:.0f}, {info['g']:.0f}, {info['blue']:.0f}), "
          f"Hue={info['hue']:.1f}, Sat={info['sat']:.1f}, ExG={info['exg']:.1f}, WAI={info['wai']:.3f}")

class_mask, cluster_map, _ = assign_land_cover_labels(labels, centers)
print("\n--- Current Label Assignments ---")
for k, v in cluster_map.items():
    print(f"Cluster {k} -> {v}")
