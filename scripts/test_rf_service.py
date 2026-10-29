import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.app.processing.preprocessing import load_image, preprocess_image
from backend.app.processing.random_forest import classify_random_forest
from backend.app.core.config import settings

img_path = r"C:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images\M-34-51-C-b-2-1.tif"
img = load_image(img_path)
resized_bgr, lab_image = preprocess_image(img, target_size=settings.DEFAULT_IMAGE_SIZE)

t0 = time.time()
class_mask, colored_rgb, area_stats = classify_random_forest(resized_bgr)
elapsed = time.time() - t0

print(f"Random Forest Classification executed in {elapsed*1000:.1f}ms!")
print("\nPredicted Area Statistics (Supervised RF):")
for s in area_stats:
    print(f"  {s['class_name']}: {s['percentage']}% ({s['pixel_count']} px)")
