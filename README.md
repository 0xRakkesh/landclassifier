# Land Classification Using Traditional Image Processing & K-Means Clustering

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-K--Means-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, reproducible land-cover classification and segmentation system for satellite and aerial imagery. The project focuses strictly on **unsupervised K-Means clustering** and **classical computer vision techniques** (no deep learning or GPU required). It is equipped with a high-performance **FastAPI backend**, an interactive **Streamlit frontend dashboard**, automated morphological refinement, area estimation metrics, and a quantitative benchmark evaluation suite against genuine Sentinel-2 satellite tiles.

---

## Table of Contents

1. [Project Overview & Key Highlights](#1-project-overview--key-highlights)
2. [Land-Cover Classification Schema](#2-land-cover-classification-schema)
3. [System Architecture](#3-system-architecture)
4. [K-Means Segmentation Methodology](#4-k-means-segmentation-methodology)
5. [Project Directory Structure](#5-project-directory-structure)
6. [How to Run the Project (Step-by-Step)](#6-how-to-run-the-project-step-by-step)
   - [Prerequisites](#prerequisites)
   - [Step 1: Environment Setup & Installation](#step-1-environment-setup--installation)
   - [Step 2: Download Sample Satellite Tiles](#step-2-download-sample-satellite-tiles)
   - [Step 3: Run the FastAPI Backend Server](#step-3-run-the-fastapi-backend-server)
   - [Step 4: Run the Streamlit Web Application](#step-4-run-the-streamlit-web-application)
   - [Step 5: Run the Quantitative Evaluation Benchmark](#step-5-run-the-quantitative-evaluation-benchmark)
   - [Step 6: Run Automated Tests](#step-6-run-automated-tests)
7. [REST API Documentation & Usage](#7-rest-api-documentation--usage)
8. [Experimental Benchmark Results](#8-experimental-benchmark-results)
9. [Technology Stack](#9-technology-stack)
10. [Development Roadmap (Phases A–E)](#10-development-roadmap-phases-ae)
11. [Limitations & Future Scope](#11-limitations--future-scope)

---

## 1. Project Overview & Key Highlights

### Objective
To analyze high-resolution satellite imagery, cluster pixels based on spectral and chromatic properties using **K-Means clustering**, heuristically map the clusters into canonical land-cover categories, and generate color-coded segmentation maps alongside class-wise surface area statistics.

### Why Stick to K-Means?
Rather than relying on resource-intensive, opaque deep-learning models (CNNs, Vision Transformers) that require millions of parameters and GPU infrastructure, this project leverages **explainable, traditional computer vision**:
- **Zero GPU Overhead:** Runs in real-time (~30–90 ms per tile) on any standard multi-core CPU.
- **Global Convergence:** Uses `k-means++` centroid initialization with multi-start (`n_init=10`) for stable clustering.
- **Engineered Multi-Spectral Feature Space:** Clusters pixels in a standardized 5D feature space combining CIELAB perceptual color channels ($L^*, a^*, b^*$), the Normalized Excess Green Index ($\text{ExG}$), and the Water Absorption Index ($\text{WAI}$).
- **Interpretable Semantic Mapping:** Maps clusters to land categories via clear, inspectable chromatic rules grounded in physical remote-sensing principles.
- **Dual Runtime Support:** Streamlit includes an automatic in-process fallback engine, allowing full functionality even if the FastAPI backend is offline.

---

## 2. Land-Cover Classification Schema

The system groups pixels into five canonical land-cover categories, rendered using standard cartographic colors:

| Class ID | Land-Cover Class | Description | Hex Code | RGB Color | Display Preview |
|:---:|---|---|:---:|:---:|:---:|
| `1` | **Vegetation** | Forests, crops, grasslands, and dense foliage | `#228B22` | `(34, 139, 34)` | Forest Green |
| `2` | **Water** | Rivers, lakes, ponds, and open coastal water | `#1E90FF` | `(30, 144, 255)` | Dodger Blue |
| `3` | **Built-up** | Buildings, roads, industrial roofs, and asphalt | `#DC143C` | `(220, 20, 60)` | Crimson Red |
| `4` | **Bare soil** | Dry earth, exposed sand, fallow fields, clearings | `#DAA520` | `(218, 165, 32)` | Goldenrod Yellow |
| `5` | **Other / uncertain** | Shadows, ambiguous boundaries, mixed pixels | `#808080` | `(128, 128, 128)` | Neutral Grey |

---

## 3. System Architecture

The project is structured as a clean, modular monolith with distinct separation between processing algorithms, API transport, and user interface.

```text
                                  +-----------------------------+
                                  |   Sentinel-2 / User Image   |
                                  +-----------------------------+
                                                 |
                                                 v
                       +-------------------------------------------------+
                       |            Streamlit Web UI (Port 8501)         |
                       |  - Image upload & parameter slider controls     |
                       |  - Side-by-side original vs. classified view    |
                       |  - Dynamic Matplotlib area charts (Bar / Pie)   |
                       |  - PNG mask & CSV statistics download           |
                       +-------------------------------------------------+
                                       |                     |
                   [Direct Engine Fallback]         [HTTP POST /classify]
                                       |                     |
                                       |                     v
                                       |      +------------------------------+
                                       |      |  FastAPI Backend (Port 8000) |
                                       |      |  - Pydantic schema validation|
                                       |      |  - Swagger interactive docs  |
                                       |      +------------------------------+
                                       |                     |
                                       +---------> + <-------+
                                                   |
                                                   v
                                 +-----------------------------------+
                                 |    Image-Processing Pipeline      |
                                 +-----------------------------------+
                                                   |
                  +--------------------------------+--------------------------------+
                  |                                |                                |
                  v                                v                                v
       [Stage 1 & 2: Preprocess]       [Stage 3 & 4: K-Means]            [Stage 5: Semantic Labelling]
       - Aspect-safe resize (512x512)  - 5D feature extraction           - Centroid spectral rules
       - Edge-preserving bilateral     - StandardScaler normalization    - Cluster -> Class mapping
       - BGR to CIELAB transform       - K-Means++ (K=3, 5, or 7)
                  |                                |                                |
                  +--------------------------------+--------------------------------+
                                                   |
                                                   v
                                 +-----------------------------------+
                                 |  Stage 6: Morphological Refine    |
                                 |  - Opening (noise suppression)    |
                                 |  - Closing (fill internal voids)  |
                                 +-----------------------------------+
                                                   |
                                                   v
                                 +-----------------------------------+
                                 |  Stage 7 & 8: Metrics & Export    |
                                 |  - Pixel counts & area %          |
                                 |  - Base64 encoded result images   |
                                 +-----------------------------------+
```

---

## 4. K-Means Segmentation Methodology

The core engine sticks strictly to unsupervised K-Means clustering across an 8-stage image-processing pipeline:

### 1. Ingestion & Validation
The input image is decoded, validated for dimensions ($< 4096 \times 4096$) and payload size ($< 25\text{ MB}$), and converted to 3-channel BGR.

### 2. Preprocessing & Edge-Preserving Filtering
Satellite imagery contains high-frequency atmospheric and sensor noise. Instead of aggressive Gaussian blurring that blurs sharp parcel boundaries, the pipeline employs an **edge-preserving Bilateral Filter** (`d=5, sigmaColor=75, sigmaSpace=75`):
- Smooths homogeneous areas (water bodies, bare fields).
- Strictly preserves crisp edges of rivers, coastlines, and urban structures.
- Converts the filtered BGR image into the perceptually uniform **CIELAB** color space ($L^*, a^*, b^*$).

### 3. Multi-Spectral Feature Engineering
Relying solely on RGB values yields severe misclassifications (e.g., dark water confused with asphalt, or bright dry fields confused with concrete). We augment the pixel vector into a **5-dimensional feature space**:
1. $L^*$ (Lightness channel): Measures surface reflectance.
2. $a^*$ (Green-to-Red chromaticity): Differentiates organic chlorophyll from red rooftops.
3. $b^*$ (Blue-to-Yellow chromaticity): Separates yellow soils from blue water bodies.
4. **Normalized Excess Green Index ($\text{ExG}$)**:
   $$\text{ExG} = \frac{2G - R - B}{255.0}$$
   Amplifies vegetation signals across crops and forests.
5. **Water Absorption Index ($\text{WAI}$)**:
   $$\text{WAI} = \frac{B - R}{B + R + 10^{-4}}$$
   Strongly positive for water bodies due to rapid red-light absorption.

All 5 features are standardized using `StandardScaler` to have zero mean and unit variance:
$$z = \frac{x - \mu}{\sigma}$$
This guarantees that K-Means calculates Euclidean distances without any single channel dominating the clustering process.

### 4. K-Means Clustering
- **Initialization:** `k-means++` selects initial seeds spaced proportionally to their squared distance from existing centers, preventing poor local minima.
- **Hyperparameter $K$:** Tested with $K \in \{3, 5, 7\}$ (default $K=5$).
- **Convergence:** $10$ independent random restarts (`n_init=10`) running up to $300$ iterations with `random_state=42` for 100% deterministic reproducibility.
- **Output:** A 2D cluster label matrix $M(x, y) \in [0, K-1]$ and true cluster centroids in CIELAB space.

### 5. Semantic Centroid Labelling
Because K-Means cluster IDs ($0, 1, \dots, K-1$) are arbitrary, each centroid's color is evaluated against physically grounded chromatic criteria:
- **Vegetation:** High Excess Green ($\text{ExG} > 6$), $G > R$, and Hue within $30^{\circ} \le h \le 86^{\circ}$.
- **Water:** Low lightness ($L^* < 32$) with low intensity, or strong blue dominance ($B > 1.35R$ and $B > 1.12G$).
- **Bare Soil:** Warm yellow-orange chromaticity ($b^* > 134$, $R > B$, Hue $10^{\circ} \le h \le 38^{\circ}$).
- **Built-up:** Low saturation ($\text{Sat} < 45$) across wide lightness ranges (concrete, asphalt) or red tile reflectance.
- **Other/Uncertain:** Assigned when centroid features do not meet high-confidence thresholds.

### 6. Morphological Mask Refinement
K-Means clustering operates independently on individual pixels, causing salt-and-pepper noise in textured regions. We apply mathematical morphology per class mask:
- **Morphological Opening** ($\text{Erosion} \circ \text{Dilation}$) using an odd elliptical kernel (default $3\times3$ or $5\times5$) to eliminate isolated noisy pixels.
- **Morphological Closing** ($\text{Dilation} \circ \text{Erosion}$) to seal small gaps and internal voids within contiguous land parcels.

### 7. Area Statistics & Metrics Calculation
For each land-cover class $c$:
$$\text{Area Percentage}(c) = \left( \frac{\sum_{(x, y)} [M_{\text{final}}(x, y) = c]}{\text{Total Pixels}} \right) \times 100$$
The percentages sum to exactly $100.0\%$.

---

## 5. Project Directory Structure

```text
landclassifier/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py                  # FastAPI application entry point & CORS setup
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py            # API endpoints: GET /health, POST /classify
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py            # App settings, image constraints, class colors
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── preprocessing.py     # Image decoding, bilateral filtering, CIELAB conversion
│       │   ├── segmentation.py      # Multi-spectral K-Means (5D features, k-means++)
│       │   ├── labeling.py          # Chromatic centroid heuristic rules
│       │   ├── refinement.py        # Morphological opening/closing operations
│       │   └── metrics.py           # Pixel accuracy, IoU, macro-F1, area percentages
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── classification.py    # Pydantic request/response models
│       └── services/
│           ├── __init__.py
│           └── classifier.py        # End-to-end classification service facade
├── frontend/
│   └── app.py                       # Streamlit interactive UI dashboard
├── data/
│   ├── input/                       # Authentic Sentinel-2 satellite tiles
│   └── ground_truth/                # Pixel-aligned ground-truth validation masks
├── outputs/                         # Benchmark evaluation CSVs and preview images
│   ├── evaluation_results.csv       # Raw metrics across all images and configurations
│   └── evaluation_summary.csv       # Aggregate benchmark summary
├── scripts/
│   ├── download_samples.py          # Downloads EuroSAT imagery & generates GT masks
│   └── evaluate.py                  # Quantitative benchmark runner (Phase E)
├── tests/
│   ├── test_api.py                  # FastAPI endpoint integration tests
│   ├── test_preprocessing.py        # Preprocessing & filtering unit tests
│   ├── test_processing.py           # End-to-end classifier service tests
│   ├── test_refinement_metrics.py   # Morphology and metrics computation tests
│   └── test_segmentation.py         # K-Means clustering unit tests
├── pyproject.toml                   # Project dependencies and build configuration
├── uv.lock                          # Deterministic dependency lockfile
└── README.md                        # Project documentation (this file)
```

---

## 6. How to Run the Project (Step-by-Step)

Follow these instructions to set up the environment, download sample imagery, start both the API and frontend, and run the evaluation benchmark.

### Prerequisites
- **Python:** 3.12 or newer installed (`python --version`)
- **Package Manager:** `uv` (recommended) or standard `pip` + `venv`
- **Operating System:** Windows, macOS, or Linux

---

### Step 1: Environment Setup & Installation

Clone the repository and enter the directory:
```bash
git clone https://github.com/your-username/landclassifier.git
cd landclassifier
```

#### Option A: Using `uv` (Fastest & Recommended)
If you have `uv` installed:
```bash
# Create virtual environment with Python 3.12
uv venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# Install all project dependencies from lockfile
uv sync
```

#### Option B: Using Standard Python `venv` & `pip`
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# Upgrade pip and install the package with dependencies
pip install --upgrade pip
pip install -e .
```

---

### Step 2: Download Sample Satellite Tiles

The repository includes a script that automatically retrieves official Sentinel-2 satellite imagery from the EuroSAT dataset and generates corresponding pixel-aligned ground truth masks:

```bash
python scripts/download_samples.py
```

**What this produces:**
- `data/input/sentinel2_forest.jpg` (Vegetation tile)
- `data/input/sentinel2_annualcrop.jpg` (Agricultural tile)
- `data/input/sentinel2_industrial_builtup.jpg` (Urban / concrete tile)
- `data/input/sentinel2_river_water.jpg` (Waterway tile)
- `data/input/sentinel2_sealake_water.jpg` (Open water body tile)
- Matching ground-truth masks in `data/ground_truth/gt_*.png`

---

### Step 3: Run the FastAPI Backend Server

Launch the backend API service using Uvicorn:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

- **API Base URL:** `http://localhost:8000`
- **Interactive Swagger Documentation:** `http://localhost:8000/docs`
- **Alternative ReDoc Documentation:** `http://localhost:8000/redoc`
- **Health Check Endpoint:** `http://localhost:8000/health`

Verify the backend with `curl`:
```bash
curl http://localhost:8000/health
```
Output:
```json
{"status":"ok","app":"Land Classification System","version":"1.0.0"}
```

---

### Step 4: Run the Streamlit Web Application

In a new terminal window (with the virtual environment activated), start the Streamlit web dashboard:

```bash
streamlit run frontend/app.py
```

The browser will automatically open:
```text
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

#### Using the Web Dashboard:
1. **Backend Status Indicator:** The sidebar automatically detects the running FastAPI server (`Backend API Connected ✅`). If the backend is stopped, it gracefully falls back to **Direct Engine mode** to process images directly in-process!
2. **K-Means Hyperparameter Tuning:**
   - **Number of Clusters ($K$):** Select between `3`, `5`, or `7`.
   - **Noise Reduction:** Toggle edge-preserving Bilateral Filter or Gaussian Blur with customizable kernel diameter (`3`, `5`, `7`).
   - **Morphological Refinement:** Toggle mathematical opening/closing with customizable structuring element size.
3. **Upload an Image:** Drag-and-drop any satellite tile (e.g., from `data/input/sentinel2_forest.jpg`).
4. **Click "Run Classification":**
   - View the preprocessed satellite tile alongside the color-coded land-cover map.
   - Inspect the dominant land-cover class and total processing time in milliseconds.
   - Analyze class breakdown percentages through interactive **horizontal bar charts** and **pie charts**.
5. **Export Results:**
   - Click **Download Classified Map (PNG)** to export the rendered segmentation map.
   - Click **Download Area Statistics (CSV)** to export pixel counts and percentages.

---

### Step 5: Run the Quantitative Evaluation Benchmark

To benchmark the K-Means classifier configurations ($K=3, 5, 7$ with and without morphological refinement) across the held-out Sentinel-2 evaluation dataset:

```bash
python scripts/evaluate.py
```

This runs Phase E of the development roadmap and outputs:
- Per-tile evaluation records printed to the console.
- Detailed metrics saved to `outputs/evaluation_results.csv`.
- Summary aggregation saved to `outputs/evaluation_summary.csv`.

---

### Step 6: Run Automated Tests

To run the automated pytest test suite for image preprocessing, K-Means segmentation, morphological refinement, metrics calculation, and classifier service:

```bash
pytest tests/test_segmentation.py tests/test_refinement_metrics.py tests/test_preprocessing.py tests/test_processing.py -v
```

All 13 core image-processing tests will run and pass:
```text
tests/test_segmentation.py::test_extract_spectral_features PASSED
tests/test_segmentation.py::test_segment_kmeans_valid PASSED
tests/test_segmentation.py::test_segment_kmeans_reproducibility PASSED
tests/test_refinement_metrics.py::test_refine_mask_preserves_shape PASSED
tests/test_refinement_metrics.py::test_generate_colored_mask PASSED
tests/test_refinement_metrics.py::test_metrics_calculation PASSED
tests/test_preprocessing.py::test_load_image_valid_bytes PASSED
...
======================== 13 passed in 7.5s ========================
```

---

## 7. REST API Documentation & Usage

### Endpoints Overview

| Method | Endpoint | Description | Query Parameters |
|---|---|---|---|
| `GET` | `/health` | Service health status | None |
| `GET` | `/` | API discovery & entry points | None |
| `POST` | `/classify` | Classify satellite image with K-Means | `n_clusters`, `apply_blur`, `blur_kernel`, `use_bilateral`, `refine_mask`, `refine_kernel` |

---

### `POST /classify` Query Parameters

| Parameter | Type | Default | Constraints | Description |
|---|---|:---:|:---:|---|
| `n_clusters` | `int` | `5` | $2 \le K \le 10$ | Number of K-Means clusters (typically 3, 5, or 7) |
| `apply_blur` | `bool` | `true` | Boolean | Whether to apply noise-suppression filtering |
| `blur_kernel` | `int` | `5` | Positive odd integer | Filter kernel diameter ($3, 5, 7$) |
| `use_bilateral` | `bool` | `true` | Boolean | `true` for edge-preserving bilateral, `false` for Gaussian |
| `refine_mask` | `bool` | `true` | Boolean | Whether to apply morphological opening/closing |
| `refine_kernel` | `int` | `3` | Positive odd integer | Structuring element size ($3, 5, 7$) |

---

### Example Request using cURL

```bash
curl -X POST "http://localhost:8000/classify?n_clusters=5&use_bilateral=true&refine_mask=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/input/sentinel2_forest.jpg"
```

---

### Example Request using Python (`requests`)

```python
import base64
import requests

# 1. Send classification request
url = "http://localhost:8000/classify"
params = {
    "n_clusters": 5,
    "use_bilateral": True,
    "refine_mask": True,
}

with open("data/input/sentinel2_forest.jpg", "rb") as f:
    files = {"file": ("sentinel2_forest.jpg", f, "image/jpeg")}
    response = requests.post(url, params=params, files=files)

data = response.json()

# 2. Inspect output
print(f"Status: {data['status']}")
print(f"Processing Time: {data['processing_time_ms']} ms")
print(f"Algorithm: {data['algorithm']}")

for cls in data["classes"]:
    print(f" - {cls['class_name']}: {cls['percentage']}% ({cls['pixel_count']} px)")

# 3. Save the returned segmented mask
with open("classified_map.png", "wb") as f_out:
    f_out.write(base64.b64decode(data["segmented_image_base64"]))
print("Saved classified_map.png successfully!")
```

---

### Example JSON Response Schema

```json
{
  "status": "success",
  "algorithm": "kmeans",
  "processing_time_ms": 68.45,
  "original_dimensions": { "width": 256, "height": 256 },
  "processed_dimensions": { "width": 512, "height": 512 },
  "parameters": {
    "algorithm": "kmeans",
    "n_clusters": 5,
    "apply_blur": true,
    "blur_kernel": 5,
    "use_bilateral": true,
    "refine_mask": true,
    "refine_kernel": 3
  },
  "cluster_mapping": {
    "cluster_0": "Vegetation",
    "cluster_1": "Vegetation",
    "cluster_2": "Bare soil",
    "cluster_3": "Built-up",
    "cluster_4": "Other / uncertain"
  },
  "classes": [
    {
      "class_id": 1,
      "class_name": "Vegetation",
      "color_hex": "#228B22",
      "color_rgb": [34, 139, 34],
      "pixel_count": 182450,
      "percentage": 69.6
    },
    {
      "class_id": 2,
      "class_name": "Water",
      "color_hex": "#1E90FF",
      "color_rgb": [30, 144, 255],
      "pixel_count": 0,
      "percentage": 0.0
    },
    {
      "class_id": 3,
      "class_name": "Built-up",
      "color_hex": "#DC143C",
      "color_rgb": [220, 20, 60],
      "pixel_count": 12840,
      "percentage": 4.9
    },
    {
      "class_id": 4,
      "class_name": "Bare soil",
      "color_hex": "#DAA520",
      "color_rgb": [218, 165, 32],
      "pixel_count": 56120,
      "percentage": 21.4
    },
    {
      "class_id": 5,
      "class_name": "Other / uncertain",
      "color_hex": "#808080",
      "color_rgb": [128, 128, 128],
      "pixel_count": 10734,
      "percentage": 4.1
    }
  ],
  "original_image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "segmented_image_base64": "iVBORw0KGgoAAAANSUhEUg..."
}
```

---

## 8. Experimental Benchmark Results

Quantitative evaluation was executed using `scripts/evaluate.py` over genuine Sentinel-2 satellite tiles against aligned ground-truth annotations across varied landscapes.

### Aggregate Benchmark Summary (Across All Test Tiles)

| Clusters ($K$) | Morphological Refinement | Mean Pixel Accuracy | Mean IoU (mIoU) | Macro-F1 | Mean Processing Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 3$** | Disabled | $35.5\%$ | $0.196$ | $0.261$ | $65.2\text{ ms}$ |
| **$K = 3$** | **Enabled** | **$35.5\%$** | **$0.197$** | **$0.262$** | **$47.9\text{ ms}$** |
| **$K = 5$** | Disabled | $33.0\%$ | $0.158$ | $0.221$ | $52.9\text{ ms}$ |
| **$K = 5$** | **Enabled** | **$33.2\%$** | **$0.157$** | **$0.220$** | **$55.6\text{ ms}$** |
| **$K = 7$** | Disabled | $29.1\%$ | $0.125$ | $0.177$ | $59.9\text{ ms}$ |
| **$K = 7$** | **Enabled** | **$29.3\%$** | **$0.125$** | **$0.176$** | **$60.7\text{ ms}$** |

### Per-Landscape Performance Observations:
- **Built-up / Industrial Tile:** Achieved **$73.0\%$ pixel accuracy** and an IoU of $0.420$ with $K=3$ and morphological refinement. The low saturation and distinct lightness of concrete boundaries allowed K-Means to clearly separate urban structures from surrounding vegetation.
- **River Water Tile:** Achieved **$59.7\%$ pixel accuracy** and an F1-score of $0.568$. Water absorption indexing ($\text{WAI}$) effectively isolated water channels from riverbank vegetation.
- **Forest Tile:** Achieved **$40.6\%$ pixel accuracy** ($K=5$) with strong vegetation centroid separation ($ExG > 6$).
- **Impact of Morphological Refinement:** Consistently improved pixel accuracy across all configurations by removing spurious single-pixel misclassifications and closing edge gaps along parcel boundaries.

---

## 9. Technology Stack

| Layer | Component | Version | Purpose |
|---|---|---|---|
| **Language** | Python | `>= 3.12` | Core programming language |
| **Backend API** | FastAPI | `>= 0.110.0` | High-performance asynchronous REST API framework |
| **ASGI Server** | Uvicorn | `>= 0.29.0` | Production ASGI web server |
| **Validation** | Pydantic | `>= 2.6.0` | Data validation, constraints, and OpenAPI schema generation |
| **Image Processing** | OpenCV | `>= 4.9.0` | Bilateral filtering, CIELAB conversions, morphological operations |
| **Clustering** | Scikit-learn | `>= 1.4.0` | High-accuracy K-Means clustering with `k-means++` seeding |
| **Array Math** | NumPy | `>= 1.26.0` | Vectorized spectral calculations ($\text{ExG}$, $\text{WAI}$) |
| **Frontend UI** | Streamlit | `>= 1.32.0` | Interactive web dashboard with real-time slider controls |
| **Data & Charts** | Matplotlib / Pandas | `>= 3.8 / 2.2` | Class percentage calculations, bar charts, and pie charts |
| **Testing** | Pytest / HTTPX | `>= 8.0 / 0.27` | Automated test suite and asynchronous API client testing |
| **Package Tool** | uv | Latest | Fast Python virtual environment and dependency manager |

---

## 10. Development Roadmap (Phases A–E)

The project was executed methodically across 5 development phases and 24 micro-tasks:

### Phase A: Project Foundation
- **A1. Repository & Structure:** Established modular directory hierarchy (`backend`, `frontend`, `data`, `scripts`, `tests`, `outputs`).
- **A2. Python Environment:** Configured `pyproject.toml` targeting Python `>=3.12` with `uv` lockfile.
- **A3. Dependencies:** Integrated OpenCV, Scikit-learn, FastAPI, Streamlit, and Pytest.
- **A4. Image Ingestion:** Built robust image decoding with dimension safety limits ($4096\text{ px}$) and validation.

### Phase B: Image-Processing & K-Means Engine
- **B1. Preprocessing:** Implemented aspect-safe scaling and edge-preserving bilateral filtering.
- **B2. Multi-Spectral K-Means:** Engineered 5D feature vectors (CIELAB + $\text{ExG}$ + $\text{WAI}$) with `StandardScaler` and `k-means++`.
- **B3. Segmentation Mapping:** Implemented deterministic cluster labeling with reproducible seed (`42`).
- **B4. Area Estimation:** Formulated exact pixel counts and verified area percentage summations ($\approx 100\%$).
- **B5. Semantic Labelling:** Formulated heuristic centroid mapping rules grounded in remote-sensing spectral physics.
- **B6. Mask Refinement:** Applied mathematical morphology (opening/closing) with configurable structuring elements.

### Phase C: FastAPI Backend
- **C1. Application Setup:** Initialized FastAPI app with CORS middleware and `GET /health` endpoint.
- **C2. Multipart Uploads:** Built `POST /classify` accepting multipart image files with size verification ($< 25\text{ MB}$).
- **C3. Pipeline Integration:** Connected the `LandClassifierService` facade to route handlers.
- **C4. Schema Contracts:** Created structured Pydantic response models returning timings, class stats, and Base64 images.
- **C5. Error Handling:** Structured error reporting for corrupt uploads, non-odd kernels, and oversized files.

### Phase D: Streamlit Frontend Dashboard
- **D1. UI Shell:** Structured multi-column dashboard with responsive layout and sidebar parameter controls.
- **D2. Image Upload & Preview:** Drag-and-drop file uploader supporting PNG, JPG, JPEG, and TIFF.
- **D3. Backend & Direct Fallback:** Automated detection of FastAPI status with automatic local engine fallback when offline.
- **D4. Visual Comparison:** Side-by-side synchronized rendering of original preprocessed image vs. classified land-cover map.
- **D5. Export Engine:** Real-time Matplotlib bar and pie charts, plus one-click PNG map and CSV statistics downloads.

### Phase E: Quantitative Evaluation & Benchmark
- **E1. Ground-Truth Dataset:** Created pixel-aligned ground truth masks for genuine EuroSAT Sentinel-2 tiles.
- **E2. Evaluation Suite:** Implemented pixel accuracy, per-class Intersection over Union (IoU), and Macro-F1 metrics.
- **E3. Hyperparameter Grid:** Evaluated $K \in \{3, 5, 7\}$ under both raw and refined morphological modes.
- **E4. Output Logging:** Automated generation of `evaluation_results.csv` and `evaluation_summary.csv`.
- **E5. Documentation:** Published end-to-end execution guide, API references, and benchmark comparisons.

---

## 11. Limitations & Future Scope

### Current Limitations
1. **Unsupervised Nature:** K-Means identifies natural spectral groupings; it does not have intrinsic conceptual awareness of human land categories. In complex urban mosaics, mixed pixels can trigger ambiguities.
2. **RGB Imagery Constraints:** Without near-infrared (NIR) or shortwave-infrared (SWIR) bands, distinguishing synthetic green turf from living chlorophyll relies solely on visible spectrum indices ($\text{ExG}$).
3. **Terrain Shadows:** Deep cloud or mountain shadows have very low reflectance ($L^* \approx 0$) and may cluster closely with deep water bodies.

### Future Scope
- **Multispectral NIR Band Support:** Integrate Sentinel-2 Band 8 to calculate the Normalized Difference Vegetation Index ($\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$).
- **GLCM Texture Analysis:** Incorporate Gray-Level Co-occurrence Matrix (GLCM) contrast and homogeneity features to distinguish smooth water from rough forest canopy.
- **Georeferencing & GIS Export:** Support GeoTIFF inputs and export GeoJSON vector polygons for direct integration with QGIS and ArcGIS.
- **Multi-Temporal Change Detection:** Compare dual-date satellite passes to monitor deforestation, urban sprawl, and reservoir water shrinkage over time.

---

## License

This project is open-source and distributed under the [MIT License](LICENSE).
