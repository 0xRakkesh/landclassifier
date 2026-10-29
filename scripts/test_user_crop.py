import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.segmentation import segment_kmeans
from backend.app.processing.labeling import analyze_centroid, assign_land_cover_labels

# Load the user's uploaded screenshot
screenshot_path = r"C:\Users\rakes\.gemini\antigravity-cli\brain\c02e604f-1fd2-4389-a353-a2d3579dbd0b\.user_uploaded\uploaded_media_1793240852073.png"
img_full = cv2.imread(screenshot_path)
print("Screenshot shape:", img_full.shape)

# Crop the "Preprocessed Satellite Image" displayed on the left
# The image has two panels side by side
h, w = img_full.shape[:2]
# The left panel is approximately x in [50, 480], y in [150, 880]
# Let's inspect by saving or finding bounding box of the left image
# Or let's test directly on a crop of the left panel
left_crop = img_full[160:880, 60:480]
cv2.imwrite("scripts/debug_crop.png", left_crop)

bgr, lab = preprocess_image(left_crop, target_size=(512, 512))
labels, centers = segment_kmeans(lab, n_clusters=5)

print("\n--- Centroids for user's screenshot crop ---")
for i, c in enumerate(centers):
    info = analyze_centroid(c)
    print(f"Cluster {i}: L={info['L']:.1f}, a={info['a']:.1f}, b={info['b']:.1f}, "
          f"RGB=({info['r']:.0f}, {info['g']:.0f}, {info['blue']:.0f}), "
          f"Hue={info['hue']:.1f}, Sat={info['sat']:.1f}, ExG={info['exg']:.1f}, WAI={info['wai']:.3f}")

class_mask, cluster_map, _ = assign_land_cover_labels(labels, centers)
print("\n--- Label Assignments ---")
for k, v in cluster_map.items():
    print(f"Cluster {k} -> {v}")
