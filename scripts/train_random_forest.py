"""
Training script: Samples balanced pixels across LandCover.ai images and ground-truth masks,
extracts 8 multi-spectral, textural, and line features, and trains an explainable Random Forest.
"""
import os
import glob
import time
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATASET_DIR = r"c:\Users\rakes\OneDrive\Desktop\landcover.ai.v1"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")
MODEL_DIR = os.path.join("backend", "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_landcover.joblib")

# Directional Line Detector Kernels (DIP Lab Program 4)
k_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
k_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
k_d1 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
k_d2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

def extract_8_features(bgr_img):
    """Extract standardized 8-feature representation."""
    h, w = bgr_img.shape[:2]
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    b = bgr_img[:, :, 0].astype(np.float32)
    g = bgr_img[:, :, 1].astype(np.float32)
    r = bgr_img[:, :, 2].astype(np.float32)

    L = lab_img[:, :, 0].astype(np.float32)
    a = lab_img[:, :, 1].astype(np.float32)
    lab_b = lab_img[:, :, 2].astype(np.float32)

    # 1. Excess Green Index
    exg = (2.0 * g - r - b) / 255.0
    # 2. Water Absorption Index
    wai = (b - r) / (b + r + 1e-4)

    # 3. Sobel Edge Gradient Magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y) / 255.0

    # 4. Local Texture Roughness (9x9 local standard deviation)
    mean_int = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    texture_roughness = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int)) / 255.0

    # 5. Directional Line Response
    r_h = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_h))
    r_v = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_v))
    r_d1 = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_d1))
    r_d2 = np.abs(cv2.filter2D(gray, cv2.CV_32F, k_d2))
    line_response = np.maximum.reduce([r_h, r_v, r_d1, r_d2]) / 255.0

    features = np.column_stack([
        L.reshape(-1),
        a.reshape(-1),
        lab_b.reshape(-1),
        exg.reshape(-1),
        wai.reshape(-1),
        edge_mag.reshape(-1),
        texture_roughness.reshape(-1),
        line_response.reshape(-1),
    ])
    return features

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))

    print(f"Sampling training data across {len(img_paths)} orthophotos...")

    # Sample balanced points per class:
    # 0: Fields, 1: Building, 2: Woodland, 3: Water, 4: Road
    samples_per_class = 25000
    collected_X = {c: [] for c in range(5)}

    t0 = time.time()
    # Read across diverse images to accumulate samples
    rng = np.random.RandomState(42)
    
    for i in range(len(img_paths)):
        # Stop early once we have enough samples for all classes
        if all(len(collected_X[c]) >= samples_per_class for c in range(5)):
            break

        img = cv2.imread(img_paths[i])
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue

        # Downsample large images for rapid feature computation
        h, w = img.shape[:2]
        scale = 1024.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_sub = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_sub = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        feats = extract_8_features(img_sub)
        labels = mask_sub.reshape(-1)

        for c in range(5):
            needed = samples_per_class - len(collected_X[c])
            if needed <= 0:
                continue
            idx_c = np.where(labels == c)[0]
            if len(idx_c) > 0:
                n_take = min(needed, len(idx_c), 3000)
                chosen = rng.choice(idx_c, size=n_take, replace=False)
                collected_X[c].append(feats[chosen])

    # Assemble training dataset
    X_list = []
    y_list = []
    for c in range(5):
        if len(collected_X[c]) > 0:
            c_feats = np.vstack(collected_X[c])
            X_list.append(c_feats)
            y_list.append(np.full(len(c_feats), c, dtype=np.int32))
            print(f"  Class {c}: {len(c_feats):,} samples")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print(f"\nTotal training samples: {len(y):,} ({time.time() - t0:.1f}s)")
    print("Fitting Supervised Random Forest Classifier (50 trees, max_depth=15)...")

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    t_train = time.time()
    rf.fit(X, y)
    print(f"Model trained in {time.time() - t_train:.2f}s!")

    # In-sample validation metrics
    preds = rf.predict(X[:10000])
    acc = accuracy_score(y[:10000], preds)
    print(f"Training Validation Accuracy: {acc * 100:.2f}%")

    # Save model and feature names
    payload = {
        "model": rf,
        "feature_names": ["L", "a", "b", "exg", "wai", "edge_mag", "texture_std", "line_response"],
        "classes": {
            0: "Farm Fields / Light Forest",
            1: "Buildings / Dry Land",
            2: "Dense Forest",
            3: "Water Bodies",
            4: "Roads / Highways",
        },
        "color_map": {
            0: (34, 197, 94),     # Green (#22C55E)
            1: (255, 215, 0),    # Yellow (#FFD700)
            2: (0, 100, 0),       # Deep Green (#006400)
            3: (30, 144, 255),    # Blue (#1E90FF)
            4: (20, 20, 20),      # Black (#141414)
        }
    }
    joblib.dump(payload, MODEL_PATH, compress=3)
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Saved calibrated model to {MODEL_PATH} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    train()
