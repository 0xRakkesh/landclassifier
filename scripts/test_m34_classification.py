import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.services.classifier import LandClassifierService

img_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif"
with open(img_path, "rb") as f:
    buf = f.read()

res = LandClassifierService.classify(buf, n_clusters=5)
print("Processing time:", res["processing_time_ms"], "ms")
print("Cluster mapping:")
for k, v in res["cluster_mapping"].items():
    print(f"  {k} -> {v}")
print("\nClass Statistics:")
for c in res["classes"]:
    print(f"  {c['class_name']}: {c['percentage']}% ({c['pixel_count']} px)")
