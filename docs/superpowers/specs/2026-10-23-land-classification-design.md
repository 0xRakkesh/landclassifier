# Land Classification System - Technical Design Document

- **Date**: 2026-10-23
- **Status**: Approved (Option A: Client-Server Architecture)
- **Specification Source**: [README.md](../../../README.md)

---

## 1. System Overview & Architecture

The system implements traditional computer vision and unsupervised clustering to classify land cover in satellite imagery. The architecture is a modular monolithic system with two distinct presentation/service tiers:

1. **Processing Engine (`backend/app/processing/`)**:
   - Pure functional Python module using OpenCV, NumPy, and scikit-learn.
   - Independent of network, HTTP, and UI layers.
2. **Backend API (`backend/app/`)**:
   - FastAPI framework exposing health checks and synchronous classification endpoints.
   - Pydantic models for request/response serialization and input validation.
3. **Frontend Dashboard (`frontend/app.py`)**:
   - Streamlit interactive interface.
   - Communicates with FastAPI over HTTP (`requests`/`httpx`).
4. **Evaluation Suite (`tests/` & Phase E scripts)**:
   - Ground-truth evaluation computing IoU, Mean IoU, Pixel Accuracy, Precision, Recall, and F1.
   - Automated test suite run via `pytest`.

```
[Satellite Image]
       │
       ▼
[Streamlit Frontend] ──HTTP POST (Image + Params)──► [FastAPI Backend]
       ▲                                                    │
       │                                                    ▼
       │                                            [Processing Pipeline]
       │                                            ├─ Preprocessing (CIELAB)
       │                                            ├─ K-Means Clustering
       │                                            ├─ Heuristic Labelling
       │                                            ├─ Morphological Refinement
       │                                            └─ Area & Metric Computations
       │                                                    │
       └────────────────JSON Response + Mask Base64─────────┘
```

---

## 2. Directory Structure

```text
landclassifier/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── classification.py
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   ├── segmentation.py
│       │   ├── labeling.py
│       │   ├── refinement.py
│       │   └── metrics.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── classifier.py
│       └── api/
│           ├── __init__.py
│           └── routes.py
├── frontend/
│   └── app.py
├── tests/
│   ├── __init__.py
│   ├── test_processing.py
│   └── test_api.py
├── data/
│   ├── input/
│   └── ground_truth/
├── outputs/
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 3. Detailed Component Specifications

### 3.1 Preprocessing (`backend/app/processing/preprocessing.py`)
- Reads raw image bytes or file paths.
- Validates image dimensions, non-zero channels, and decodes to BGR.
- Resizes image to standard resolution (e.g., $512 \times 512$ default).
- Applies Gaussian Blur with configurable kernel size (default $3 \times 3$) to suppress noise.
- Converts BGR to CIELAB ($L^*a^*b^*$) color space.

### 3.2 K-Means Segmentation (`backend/app/processing/segmentation.py`)
- Reshapes image tensor from $(H, W, 3)$ to $(H \times W, 3)$ feature matrix.
- Runs `sklearn.cluster.KMeans` (or `MiniBatchKMeans`) with configurable $K$ ($K \in \{3, 5, 7\}$, default 5) and `random_state=42`.
- Outputs cluster labels $(H, W)$ and cluster centroid coordinates in CIELAB space.

### 3.3 Semantic Land-Cover Labelling (`backend/app/processing/labeling.py`)
- Maps cluster centroids to five classes:
  1. **Vegetation** (Green: RGB `[34, 139, 34]`): Marked by negative $a^*$ (green chroma) and moderate $L^*$.
  2. **Water bodies** (Blue: RGB `[30, 144, 255]`): Marked by low lightness $L^*$ and negative or low $b^*$.
  3. **Built-up areas** (Red: RGB `[220, 20, 60]`): Marked by higher $L^*$ and neutral/gray chromas or specific urban signatures.
  4. **Bare soil** (Yellow: RGB `[218, 165, 32]`): Marked by high $b^*$ (yellow-brown) and moderate $L^*$.
  5. **Other / Uncertain** (Grey: RGB `[128, 128, 128]`): Assigned to residual/ambiguous clusters.
- Generates a colored segmentation mask $(H, W, 3)$ using the standard palette.

### 3.4 Mask Refinement (`backend/app/processing/refinement.py`)
- Applies morphological opening (`cv2.MORPH_OPEN`) using an elliptical kernel to eliminate salt-and-pepper noise.
- Applies morphological closing (`cv2.MORPH_CLOSE`) to seal small holes.
- Preserves mask dimensions and category IDs.

### 3.5 Metrics & Statistics (`backend/app/processing/metrics.py`)
- Calculates pixel counts and area percentages per land-cover category:
  $$\text{Area\%}(c) = \frac{\text{count}(c)}{H \times W} \times 100$$
- Computes supervised evaluation metrics given ground-truth masks:
  - Pixel Accuracy
  - Per-class Precision, Recall, F1-Score
  - Per-class IoU: $\text{IoU}(c) = \frac{\text{TP}(c)}{\text{TP}(c) + \text{FP}(c) + \text{FN}(c)}$
  - Mean IoU ($\text{mIoU}$)

### 3.6 FastAPI Backend (`backend/app/main.py` & `routes.py`)
- `GET /health`: Returns service health status and API version.
- `POST /classify`:
  - Multipart upload: `file: UploadFile`
  - Query parameters: `n_clusters: int = 5`, `apply_blur: bool = True`, `blur_kernel: int = 3`, `refine_mask: bool = True`
  - Returns JSON schema:
    - `status`: string
    - `processing_time_ms`: float
    - `image_dimensions`: `{"width": int, "height": int}`
    - `parameters`: dict
    - `classes`: list of `{"class_id": int, "class_name": str, "color_hex": str, "pixel_count": int, "percentage": float}`
    - `segmented_image_base64`: Base64-encoded PNG of the classified color map.
    - `original_image_base64`: Base64-encoded PNG of the preprocessed image.

### 3.7 Streamlit Frontend (`frontend/app.py`)
- Header and documentation sidebar.
- File uploader supporting `.png`, `.jpg`, `.jpeg`, `.tif`.
- Controls for $K$ ($3, 5, 7$), Gaussian blur toggle/slider, and morphological refinement toggle.
- Side-by-side display of original satellite image and segmented land-cover map.
- Metric cards showing dominant land-cover class.
- Interactive Matplotlib bar and pie charts for area percentage breakdown.
- Download buttons for classified image (PNG) and statistics table (CSV).

---

## 4. Testing & Evaluation Plan

1. **Automated Unit Tests (`tests/test_processing.py`)**:
   - Verification of image preprocessing output shapes and types.
   - Verification that K-means produces $K$ clusters and labels sum to 100%.
   - Verification of morphological refinement stability.
   - Verification of IoU and confusion matrix calculation correctness.
2. **Automated API Tests (`tests/test_api.py`)**:
   - `GET /health` returns 200 OK.
   - `POST /classify` processes valid images and returns the expected schema.
   - Validation handling for corrupted/non-image files.
3. **Phase E Evaluation**:
   - Benchmark performance across $K \in \{3, 5, 7\}$ on sample satellite images with ground-truth masks.
   - Save comparison metrics table to `outputs/evaluation_results.csv`.
