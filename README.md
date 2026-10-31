# LandClassifier

## A practical guide to satellite land-cover classification

LandClassifier turns an aerial or satellite image into a semantic land-cover map. Instead of returning only a label for an entire image, it classifies pixels and produces a color-coded mask showing vegetation, water, built-up areas, roads, and uncertain regions.

The project is intentionally built from explainable computer-vision and classical machine-learning components:

1. The input image is decoded, validated, resized, and optionally denoised.
2. Eight spectral, textural, edge, and directional features are calculated for every pixel.
3. A supervised Random Forest model predicts the LandCover.ai class for each pixel.
4. Dataset labels are translated into the application's semantic classes.
5. Optional morphological refinement removes isolated noise and fills small gaps.
6. The result is returned as a PNG mask together with coverage statistics and optional evaluation metrics.

The repository includes both a browser interface and an HTTP API:

| Interface | Entry point | Best for |
| --- | --- | --- |
| Streamlit | `frontend/app.py` | Interactive local analysis |
| Gradio | `app.py` | A shareable demo or hosted application |
| FastAPI | `backend.app.main:app` | Scripts, services, and batch workflows |

## Table of contents

- [What the classifier recognizes](#what-the-classifier-recognizes)
- [How the pipeline works](#how-the-pipeline-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model weights](#model-weights)
- [Running the applications](#running-the-applications)
- [Using the REST API](#using-the-rest-api)
- [Understanding the output](#understanding-the-output)
- [Feature engineering](#feature-engineering)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Development and testing](#development-and-testing)
- [Repository guide](#repository-guide)
- [Troubleshooting](#troubleshooting)
- [Limitations and roadmap](#limitations-and-roadmap)
- [License](#license)

## What the classifier recognizes

The output uses a fixed application palette. Roads and buildings intentionally share the same yellow visualization color, because both represent impervious or developed land in the final map.

| Application ID | Class | Color | Meaning |
| ---: | --- | --- | --- |
| 1 | Dense Forest | `#006400` | Dense canopy and thick forest |
| 2 | Farm Fields / Light Forest | `#22C55E` | Crops, fields, pastures, and lighter vegetation |
| 3 | Water Bodies | `#1E90FF` | Rivers, lakes, reservoirs, ponds, and canals |
| 4 | Roads / Highways | `#FFD700` | Roads, highways, paved paths, and runways |
| 5 | Buildings / Dry Land | `#FFD700` | Buildings, roofs, concrete, bare soil, and sand |
| 6 | Unclassified | `#808080` | Shadows, mixed terrain, or ambiguous regions |

Although classes 4 and 5 retain separate IDs in the configuration, both are rendered with `#FFD700`. The trained model maps the LandCover.ai road label into the same application output group as buildings and dry land.

## How the pipeline works

### 1. Image preparation

Images are read from bytes with OpenCV. The original dimensions are retained for reporting, while inference uses the configured processing size of `512 × 512` pixels. A bilateral filter can be applied to reduce noise while preserving boundaries.

### 2. Feature extraction

For each processed pixel, the classifier creates an eight-value feature vector:

| Feature | Purpose |
| --- | --- |
| CIELAB `L*` | Separates bright and dark surfaces |
| CIELAB `a*` | Captures the green-to-red color axis |
| CIELAB `b*` | Captures the blue-to-yellow color axis |
| Excess Green (`ExG`) | Highlights vegetation in RGB imagery |
| Water Absorption Index (`WAI`) | Uses blue/red contrast to support water detection |
| Sobel edge magnitude | Identifies boundaries and structural edges |
| `9 × 9` texture roughness | Separates rough canopy from smooth surfaces |
| Directional line response | Detects linear roads and transport corridors |

### 3. Random Forest prediction

The model predicts LandCover.ai labels for all pixels. Those dataset labels are mapped to the application IDs, then optionally refined with a morphological operation. The trained model is loaded lazily and cached after the first request.

### 4. Statistics and evaluation

The service counts pixels in every class and calculates coverage percentages. If a ground-truth mask is supplied, it also returns pixel accuracy, macro precision, macro recall, macro F1, mean IoU, and per-class metrics.

## Requirements

- Python 3.12 or newer
- The dependencies in `pyproject.toml` or `requirements.txt`
- A trained model file, unless `MODEL_DOWNLOAD_URL` is configured

[`uv`](https://docs.astral.sh/uv/) is recommended because the repository includes `uv.lock`, but standard `pip` is also supported.

## Installation

```bash
git clone https://github.com/0xRakkesh/landclassifier.git
cd landclassifier
uv sync
```

With `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Model weights

The model is deliberately excluded from Git because it is a large binary. The classifier expects:

```text
backend/app/models/random_forest_landcover.joblib
```

Place the trained file at that path before starting an interface. For hosted deployments, provide a URL through `MODEL_DOWNLOAD_URL`; the service downloads the file on the first inference request:

```bash
export MODEL_DOWNLOAD_URL="https://example.com/random_forest_landcover.joblib"
```

The URL should point to a trusted, accessible model artifact. If neither the local file nor the environment variable is available, classification fails with a clear `FileNotFoundError`.

The repository does include the small benchmark file:

```text
backend/app/models/model_metrics.json
```

It is used by the user interfaces and the `/metrics` endpoint.

## Running the applications

### Streamlit

Streamlit is the simplest local workflow:

```bash
uv run streamlit run frontend/app.py --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501), upload a PNG, JPEG, or TIFF image, and select **Classify Land Cover**. The page shows the processed image, the color mask, inference time, and class coverage.

### Gradio

The root `app.py` provides a second interactive interface:

```bash
uv run python app.py
```

It listens on port `7860` and includes example images when the files in `examples/` are present.

### FastAPI

Start the API with:

```bash
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Useful URLs:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Benchmark metrics: [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

## Using the REST API

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "app": "Land Classification System",
  "version": "2.2.0"
}
```

### Classify an image

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -F "file=@examples/sample_forest_field.png"
```

The endpoint accepts these query parameters:

| Parameter | Default | Description |
| --- | ---: | --- |
| `apply_blur` | `true` | Enable preprocessing blur |
| `blur_kernel` | `5` | Odd blur kernel size |
| `use_bilateral` | `true` | Preserve edges during denoising |
| `refine_mask` | `true` | Apply morphological refinement |
| `refine_kernel` | `3` | Odd refinement kernel size |
| `mask_file` | absent | Optional ground-truth mask for live evaluation |

For example:

```bash
curl -X POST \
  "http://127.0.0.1:8000/classify?apply_blur=true&blur_kernel=5&use_bilateral=true&refine_mask=true&refine_kernel=3" \
  -F "file=@examples/sample_roads_urban.png" \
  -F "mask_file=@path/to/ground-truth-mask.png"
```

`blur_kernel` and `refine_kernel` must be positive odd integers. The API returns HTTP 400 for even values or empty uploads.

### Response shape

The JSON response contains:

```text
status
algorithm
processing_time_ms
original_dimensions
processed_dimensions
parameters
cluster_mapping
classes
original_image_base64
segmented_image_base64
evaluation
```

The two image fields contain base64-encoded PNG data. Each item in `classes` includes `class_id`, `class_name`, `color_hex`, `color_rgb`, `pixel_count`, and `percentage`. `evaluation.has_ground_truth` is `false` unless a usable mask was supplied or automatically found.

## Understanding the output

The classifier always reports both original and processed dimensions. This distinction matters because a large input image is resized to the configured inference size before feature extraction.

Coverage is calculated as:

```text
percentage = class_pixel_count / (processed_width × processed_height) × 100
```

Metrics are percentages rather than fractions. For example, an accuracy of `75.63` means `75.63%`.

When a file name is passed to the service, it also checks the local `data/masks/` directory for a mask with the same base file name. Supplying `mask_file` to the API is more portable and is recommended for deployments.

## Configuration

Application defaults live in `backend/app/core/config.py`:

| Setting | Default |
| --- | --- |
| Project name | `Land Classification System` |
| API version | `2.2.0` |
| Processing size | `512 × 512` |
| Maximum image dimension | `16384` |
| Maximum file size setting | `100 MB` |
| Supported K-means values | `3`, `5`, `7` |

The service primarily uses the supervised Random Forest path. K-means and heuristic labeling remain available in the processing modules for compatibility and experimentation.

## Evaluation

Offline benchmark values are stored in `backend/app/models/model_metrics.json`. The same values are displayed by the web interfaces and returned by `GET /metrics`.

For a live evaluation, provide a ground-truth mask to `/classify`. The service resizes the mask with nearest-neighbor interpolation, maps LandCover.ai IDs to application IDs, and compares it with the predicted mask.

The repository also contains experiment scripts for training, comparison, inspection, and evaluation under `scripts/`. These scripts may require the original LandCover.ai dataset, which is not committed to the repository.

## Development and testing

Run the test suite with:

```bash
uv run pytest
```

The tests cover preprocessing, feature extraction, segmentation, refinement, metrics, and API behavior. To run a focused group:

```bash
uv run pytest tests/test_api.py tests/test_processing.py
```

Keep generated datasets, model binaries, caches, and evaluation outputs out of commits. The repository `.gitignore` already excludes those artifacts.

## Repository guide

```text
landclassifier/
├── app.py                         # Gradio application
├── frontend/app.py                # Streamlit application
├── backend/app/
│   ├── main.py                    # FastAPI application
│   ├── api/routes.py              # HTTP endpoints
│   ├── core/config.py             # Defaults and class palette
│   ├── schemas/classification.py  # Pydantic response models
│   ├── services/classifier.py     # End-to-end orchestration
│   ├── processing/
│   │   ├── preprocessing.py       # Decode, resize, and denoise
│   │   ├── random_forest.py       # Feature extraction and RF inference
│   │   ├── segmentation.py        # K-means segmentation
│   │   ├── labeling.py            # Cluster-to-class labeling
│   │   ├── refinement.py          # Mask cleanup and color rendering
│   │   └── metrics.py             # Area and evaluation metrics
│   └── models/
│       └── model_metrics.json     # Offline benchmark metadata
├── examples/                      # Small demonstration images
├── scripts/                       # Training and evaluation utilities
├── tests/                         # Automated tests
├── docs/                          # Design and project planning documents
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # pip-compatible dependencies
└── uv.lock                        # Reproducible uv resolution
```

## Troubleshooting

### `random_forest_landcover.joblib` is missing

Install the model artifact at `backend/app/models/random_forest_landcover.joblib`, or set `MODEL_DOWNLOAD_URL` before starting the application.

### The API rejects a kernel parameter

Both `blur_kernel` and `refine_kernel` must be odd. Use values such as `3`, `5`, or `7`.

### The image is large or slow to process

Inference is performed at `512 × 512`, but decoding a very large source image still consumes memory. Use a smaller tile where possible, and keep source dimensions below the configured maximum.

### The mask colors do not distinguish roads from buildings

This is intentional. Both application classes use `#FFD700` in the final visualization. Inspect `class_id` values in the API response when the distinction is needed programmatically.

## Limitations and roadmap

The classifier is designed for practical, explainable RGB orthophoto classification rather than universal remote-sensing segmentation. Accuracy depends on image resolution, lighting, geography, and similarity to the training data. The current pipeline does not use multispectral satellite bands beyond the information available in the input image.

Potential future improvements include confidence thresholds, connected-component filtering, richer spatial context, additional training data, and deep-learning models such as U-Net or SegFormer.

## License

This project is released under the MIT License. See the repository for the complete license text and attribution details.
