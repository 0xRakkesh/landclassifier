"""
Supervised Random Forest Training & Rigorous Test-Set Evaluation.
Trained on: data/train.txt (80% split)
Evaluated on: data/test.txt (20% split)
Calculates: Accuracy, Precision, Recall, IoU, and F1-Score.
"""
import os
import time
import json
import random
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMGS_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_landcover.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# Directional Line Detector Kernels
K_H = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
K_V = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
K_D1 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
K_D2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

CLASS_NAMES = {
    0: "Farm Fields / Light Forest",
    1: "Buildings / Dry Land",
    2: "Dense Forest",
    3: "Water Bodies",
    4: "Roads / Highways",
}

def extract_features(bgr_img: np.ndarray) -> np.ndarray:
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    b = bgr_img[:, :, 0].astype(np.float32)
    g = bgr_img[:, :, 1].astype(np.float32)
    r = bgr_img[:, :, 2].astype(np.float32)

    L = lab_img[:, :, 0].astype(np.float32)
    a = lab_img[:, :, 1].astype(np.float32)
    lab_b = lab_img[:, :, 2].astype(np.float32)

    exg = (2.0 * g - r - b) / 255.0
    wai = (b - r) / (b + r + 1e-4)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y) / 255.0

    mean_int = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    texture_roughness = np.sqrt(np.maximum(0, mean_sq - mean_int * mean_int)) / 255.0

    r_h = np.abs(cv2.filter2D(gray, cv2.CV_32F, K_H))
    r_v = np.abs(cv2.filter2D(gray, cv2.CV_32F, K_V))
    r_d1 = np.abs(cv2.filter2D(gray, cv2.CV_32F, K_D1))
    r_d2 = np.abs(cv2.filter2D(gray, cv2.CV_32F, K_D2))
    line_response = np.maximum.reduce([r_h, r_v, r_d1, r_d2]) / 255.0

    return np.column_stack([
        L.reshape(-1),
        a.reshape(-1),
        lab_b.reshape(-1),
        exg.reshape(-1),
        wai.reshape(-1),
        edge_mag.reshape(-1),
        texture_roughness.reshape(-1),
        line_response.reshape(-1),
    ])

def run_train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(os.path.join(DATA_DIR, "test.txt")) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    print(f"=== Supervised Random Forest Training Pipeline ===")
    print(f"Train set: {len(train_ids)} tiles | Test set: {len(test_ids)} tiles")

    # Sample balanced training pixels across train tiles
    print("\n--- 1. Sampling Balanced Multi-Spectral Training Data ---")
    samples_per_class = 60000
    collected_X = {c: [] for c in range(5)}
    rng = np.random.RandomState(42)

    # Shuffle training tiles to draw samples uniformly
    sample_pool = train_ids.copy()
    rng.shuffle(sample_pool)

    t0 = time.time()
    for tile_id in sample_pool:
        if all(len(collected_X[c]) >= samples_per_class for c in range(5)):
            break

        img_p = os.path.join(IMGS_DIR, f"{tile_id}.png")
        mask_p = os.path.join(MASKS_DIR, f"{tile_id}.png")
        if not os.path.exists(img_p) or not os.path.exists(mask_p):
            continue

        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue

        feats = extract_features(img)
        labels = mask.reshape(-1)

        for c in range(5):
            needed = samples_per_class - len(collected_X[c])
            if needed <= 0:
                continue
            idx_c = np.where(labels == c)[0]
            if len(idx_c) > 0:
                n_take = min(needed, len(idx_c), 1500)
                chosen = rng.choice(idx_c, size=n_take, replace=False)
                collected_X[c].append(feats[chosen])

    X_list, y_list = [], []
    for c in range(5):
        c_feats = np.vstack(collected_X[c])
        X_list.append(c_feats)
        y_list.append(np.full(len(c_feats), c, dtype=np.int32))
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(c_feats):,} samples")

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    perm = rng.permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print(f"\nTotal Training Samples: {len(y_train):,} pixels extracted in {time.time() - t0:.1f}s")
    print("--- 2. Fitting Random Forest Model (100 Trees, max_depth=18) ---")
    t_fit = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"Random Forest fitted in {time.time() - t_fit:.2f}s!")

    # Rigorous Evaluation on the Unseen Test Set
    print("\n--- 3. Evaluating on Unseen Test Dataset (data/test.txt) ---")
    test_eval_pool = test_ids.copy()
    rng.shuffle(test_eval_pool)
    eval_tiles = test_eval_pool[:200]  # Evaluate across 200 diverse test tiles (50 million pixels)

    y_true_all = []
    y_pred_all = []
    class_intersections = {c: 0 for c in range(5)}
    class_unions = {c: 0 for c in range(5)}

    t_eval = time.time()
    for tile_id in eval_tiles:
        img_p = os.path.join(IMGS_DIR, f"{tile_id}.png")
        mask_p = os.path.join(MASKS_DIR, f"{tile_id}.png")
        if not os.path.exists(img_p) or not os.path.exists(mask_p):
            continue

        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue

        feats = extract_features(img)
        preds = rf.predict(feats)
        truth = mask.reshape(-1)

        y_true_all.append(truth)
        y_pred_all.append(preds)

        for c in range(5):
            p_c = (preds == c)
            t_c = (truth == c)
            class_intersections[c] += np.sum(p_c & t_c)
            class_unions[c] += np.sum(p_c | t_c)

    y_true_flat = np.concatenate(y_true_all)
    y_pred_flat = np.concatenate(y_pred_all)

    # Compute Metrics
    overall_accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, labels=list(range(5)), zero_division=0
    )

    per_class_iou = {}
    valid_ious = []
    for c in range(5):
        if class_unions[c] > 0:
            iou_val = float(class_intersections[c] / class_unions[c])
            per_class_iou[c] = round(iou_val * 100.0, 2)
            valid_ious.append(iou_val)
        else:
            per_class_iou[c] = 0.0

    mean_iou = round(float(np.mean(valid_ious) * 100.0), 2)
    macro_precision = round(float(np.mean(precision) * 100.0), 2)
    macro_recall = round(float(np.mean(recall) * 100.0), 2)
    macro_f1 = round(float(np.mean(f1) * 100.0), 2)
    overall_acc_pct = round(float(overall_accuracy * 100.0), 2)

    print(f"\n================ TEST BENCHMARK RESULTS ================")
    print(f"Overall Accuracy:  {overall_acc_pct}%")
    print(f"Macro Precision:   {macro_precision}%")
    print(f"Macro Recall:      {macro_recall}%")
    print(f"Macro F1-Score:    {macro_f1}%")
    print(f"Mean IoU (mIoU):   {mean_iou}%")
    print(f"Evaluation Time:   {time.time() - t_eval:.1f}s over {len(y_true_flat):,} pixels")
    print(f"---------------------------------------------------------")
    print(f"{'Class':<28} | {'Precision':<10} | {'Recall':<10} | {'IoU':<10} | {'Samples':<10}")
    print(f"{'-'*28}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    per_class_metrics = {}
    for c in range(5):
        c_name = CLASS_NAMES[c]
        p_pct = round(float(precision[c] * 100.0), 2)
        r_pct = round(float(recall[c] * 100.0), 2)
        iou_pct = per_class_iou[c]
        samp = int(support[c])
        print(f"{c_name:<28} | {p_pct:>8.2f}% | {r_pct:>8.2f}% | {iou_pct:>8.2f}% | {samp:>10,}")
        per_class_metrics[c_name] = {
            "class_id": c,
            "precision": p_pct,
            "recall": r_pct,
            "iou": iou_pct,
            "support": samp,
        }

    # Save model and metrics payload
    payload = {
        "model": rf,
        "feature_names": ["L", "a", "b", "exg", "wai", "edge_mag", "texture_std", "line_response"],
        "classes": CLASS_NAMES,
        "benchmark": {
            "overall_accuracy": overall_acc_pct,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "mean_iou": mean_iou,
            "per_class": per_class_metrics,
            "test_tiles_evaluated": len(eval_tiles),
            "pixels_evaluated": int(len(y_true_flat)),
        }
    }
    joblib.dump(payload, MODEL_PATH, compress=3)
    print(f"\nSaved calibrated Random Forest model to: {MODEL_PATH}")

    # Write metrics to JSON for frontend display
    metrics_summary = {
        "model_name": "Supervised Random Forest (LandCover.ai)",
        "train_set_size": len(train_ids),
        "test_set_size": len(test_ids),
        "overall_accuracy": overall_acc_pct,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mean_iou": mean_iou,
        "per_class": per_class_metrics,
        "features": ["CIELAB (L, a, b)", "Excess Green Index (ExG)", "Water Absorption Index (WAI)", "Sobel Gradient Magnitude", "9x9 Texture Roughness", "Directional Line Detectors"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved benchmark metrics JSON to: {METRICS_PATH}")

if __name__ == "__main__":
    run_train_and_evaluate()
