# Land Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, reproducible land-cover classification system using traditional image processing (OpenCV, K-Means clustering in CIELAB space, morphological refinement), a FastAPI REST backend, a Streamlit interactive frontend, and comprehensive evaluation on real satellite imagery.

**Architecture:** A modular client-server architecture. The image processing engine (`backend/app/processing/`) provides pure functions for image transformation, clustering, heuristic semantic labeling, refinement, and area metrics. A FastAPI backend exposes `GET /health` and `POST /classify` endpoints, and a Streamlit UI enables users to upload satellite images, tune hyperparameters ($K$, blur, morphology), inspect classified maps and area percentages, and download CSV reports.

**Tech Stack:** Python 3.12, `uv`, FastAPI, Uvicorn, OpenCV (`opencv-python-headless`), NumPy, scikit-learn, Streamlit, Matplotlib, pandas, pytest, HTTPX.

**Spec:** `docs/superpowers/specs/2026-10-23-land-classification-design.md`

## Global Constraints
- Target Python version: 3.12 managed via `uv`.
- Directory structure strictly matching README Section 7.
- Color palette: Vegetation (Green `#228B22`), Water (`#1E90FF`), Built-up (`#DC143C`), Bare soil (`#DAA520`), Uncertain (`#808080`).
- Clustering must use reproducible seed (`random_state=42`).
- All tests must pass via `uv run pytest`.

---

### Task 1: Environment & Project Foundation (Phase A)

**Files:**
- Create: `pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `data/input/.gitkeep`
- Create: `data/ground_truth/.gitkeep`
- Create: `outputs/.gitkeep`

**Interfaces:**
- Produces: `backend.app.core.config.Settings` with configuration defaults (e.g. `DEFAULT_K=5`, `DEFAULT_BLUR_KERNEL=3`, `DEFAULT_IMAGE_SIZE=(512, 512)`).

- [x] **Step 1: Write `pyproject.toml` with required dependencies**
- [x] **Step 2: Create directory tree and `.gitkeep` files**
- [x] **Step 3: Implement `backend/app/core/config.py`**
- [x] **Step 4: Initialize virtual environment and lockfile with `uv sync` or `uv run`**
- [x] **Step 5: Verify environment by testing imports (`fastapi`, `cv2`, `sklearn`, `streamlit`)**

---

### Task 2: Processing Engine — Preprocessing (Phase B1, A4)

**Files:**
- Create: `backend/app/processing/__init__.py`
- Create: `backend/app/processing/preprocessing.py`
- Create: `tests/__init__.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Produces:
  - `load_image(image_input: bytes | str | Path) -> np.ndarray`: loads and validates BGR image.
  - `preprocess_image(image_bgr: np.ndarray, target_size: tuple[int, int] = (512, 512), apply_blur: bool = True, blur_kernel: int = 3) -> tuple[np.ndarray, np.ndarray]`: returns `(resized_bgr, lab_image)`.

- [x] **Step 1: Write unit tests in `tests/test_preprocessing.py` for valid/invalid images, resizing, and CIELAB conversion**
- [x] **Step 2: Run test to verify failure (`uv run pytest tests/test_preprocessing.py`)**
- [x] **Step 3: Implement `backend/app/processing/preprocessing.py`**
- [x] **Step 4: Run test to verify it passes**

---

### Task 3: Processing Engine — Segmentation & Semantic Labelling (Phase B2, B3, B4, B5)

**Files:**
- Create: `backend/app/processing/segmentation.py`
- Create: `backend/app/processing/labeling.py`
- Create: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: `lab_image: np.ndarray` from `preprocessing.py`.
- Produces:
  - `segment_kmeans(lab_image: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]`: returns `(cluster_labels_2d, cluster_centers_lab)`.
  - `assign_land_cover_labels(cluster_labels_2d: np.ndarray, cluster_centers_lab: np.ndarray) -> tuple[np.ndarray, dict[int, str], np.ndarray]`: returns `(class_mask_2d, cluster_to_class_map, colored_mask_rgb)`.

- [x] **Step 1: Write unit tests in `tests/test_segmentation.py` verifying cluster shape, centroid mapping, and palette output**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Implement `backend/app/processing/segmentation.py`**
- [x] **Step 4: Implement `backend/app/processing/labeling.py` with CIELAB heuristic land-cover rules**
- [x] **Step 5: Run tests and ensure 100% pass**

---

### Task 4: Processing Engine — Refinement & Area Metrics (Phase B6, Stage 7, Stage 9)

**Files:**
- Create: `backend/app/processing/refinement.py`
- Create: `backend/app/processing/metrics.py`
- Create: `tests/test_refinement_metrics.py`

**Interfaces:**
- Consumes: `class_mask_2d: np.ndarray` from `labeling.py`.
- Produces:
  - `refine_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray`: morphological opening and closing.
  - `calculate_area_statistics(mask: np.ndarray) -> list[dict]`: returns pixel count and percentage per category.
  - `compute_segmentation_metrics(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray, class_ids: list[int]) -> dict`: returns Pixel Accuracy, Per-class Precision/Recall/F1/IoU, and Mean IoU.

- [x] **Step 1: Write unit tests for morphological cleaning and exact metric calculations**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Implement `refinement.py` and `metrics.py`**
- [x] **Step 4: Run test to verify it passes**

---

### Task 5: Classifier Service Facade & Integrated Processing Tests (Phase B Final)

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/classifier.py`
- Create: `tests/test_processing.py`

**Interfaces:**
- Produces:
  - `LandClassifierService.classify(image_bytes: bytes, n_clusters: int, apply_blur: bool, blur_kernel: int, refine_mask: bool) -> ClassificationResult`
  - Encodes images to base64 strings and computes timing.

- [x] **Step 1: Write integration tests in `tests/test_processing.py`**
- [x] **Step 2: Implement `LandClassifierService` in `backend/app/services/classifier.py`**
- [x] **Step 3: Run `uv run pytest tests/test_processing.py` to confirm full pipeline passes**

---

### Task 6: FastAPI Backend & API Tests (Phase C)

**Files:**
- Create: `backend/app/schemas/__init__.py`
- Create: `backend/app/schemas/classification.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes.py`
- Create: `backend/app/main.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Endpoints:
  - `GET /health` -> `{"status": "ok", "app": "Land Cover Classifier", "version": "1.0.0"}`
  - `POST /classify` -> `ClassificationResponse`

- [x] **Step 1: Write API tests in `tests/test_api.py` for `/health`, `/classify` with valid image, and invalid file upload**
- [x] **Step 2: Implement Pydantic schemas in `schemas/classification.py`**
- [x] **Step 3: Implement route handlers in `api/routes.py` and FastAPI application in `main.py`**
- [x] **Step 4: Run `uv run pytest tests/test_api.py`**

---

### Task 7: Streamlit Frontend Application (Phase D)

**Files:**
- Create: `frontend/app.py`

**Interfaces:**
- Consumes: Backend API `http://localhost:8000/classify` and `http://localhost:8000/health`.
- Produces: Interactive dashboard with file upload, sidebar controls, side-by-side images, Matplotlib charts, and CSV download.

- [x] **Step 1: Build the Streamlit UI with sidebar settings ($K$, blur, morphology toggles)**
- [x] **Step 2: Add API communication logic with fallback demo mode if backend is starting**
- [x] **Step 3: Add Matplotlib visualization for class area percentages**
- [x] **Step 4: Add CSV and PNG export buttons**
- [x] **Step 5: Verify Streamlit syntax and startup headless check**

---

### Task 8: Real Satellite Data, Ground Truth & Parameter Evaluation (Phase E)

**Files:**
- Create: `scripts/download_samples.py`
- Create: `scripts/evaluate.py`
- Populate: `data/input/` with genuine satellite images
- Populate: `data/ground_truth/` with corresponding ground truth masks
- Output: `outputs/evaluation_results.csv` and comparison figures

**Interfaces:**
- Produces: Quantitative evaluation table across $K \in \{3, 5, 7\}$ testing IoU, Pixel Accuracy, Precision, Recall, and F1 on real satellite imagery.

- [x] **Step 1: Implement `scripts/download_samples.py` to fetch genuine satellite tiles (EuroSAT / Sentinel-2 samples)**
- [x] **Step 2: Provide/generate aligned ground-truth masks for evaluation**
- [x] **Step 3: Implement `scripts/evaluate.py` comparing $K=3, 5, 7$ with/without morphological refinement**
- [x] **Step 4: Execute evaluation and verify outputs saved to `outputs/evaluation_results.csv`**
- [x] **Step 5: Run full test suite: `uv run pytest`**
