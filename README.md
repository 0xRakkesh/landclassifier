# 🛰️ Satellite Land-Cover Classification System

An industrial-grade, explainable machine learning system for satellite and aerial orthophoto segmentation using **Supervised Random Forest** calibrated on the LandCover.ai dataset.

---

## 🌟 Key Highlights

- **Dataset**: Converted into **11,366 non-overlapping $500 \times 500$ PNG tiles** (`data/images/` and `data/masks/`).
- **80 / 20 Partition**:
  - `data/train.txt`: **9,092 tiles** (80.0%)
  - `data/test.txt`: **2,274 tiles** (20.0%)
- **Supervised Random Forest**: 100 decision trees trained on balanced multi-spectral and textural features.
- **8 Extracted Features**:
  1. $L^*$ (CIELAB Lightness)
  2. $a^*$ (Green–Red chromatic axis)
  3. $b^*$ (Blue–Yellow chromatic axis)
  4. $ExG$ (Excess Green Index: $2G - R - B$)
  5. $WAI$ (Water Absorption Index: $\frac{B - R}{B + R}$)
  6. Sobel Edge Gradient Magnitude
  7. $9 \times 9$ Local Texture Roughness (Standard Deviation)
  8. Directional Line Detectors ($K_H, K_V, K_{D1}, K_{D2}$)
- **Rigorous Evaluation on Test Set (`data/test.txt`)**:
  - **Overall Accuracy**: **75.63%**
  - **Macro Precision**: **59.48%**
  - **Macro Recall**: **68.55%**
  - **Macro F1-Score**: **61.41%**
  - **Mean IoU (mIoU)**: **46.91%**

| Land-Cover Class | Precision | Recall | IoU | Test Pixel Support |
|:---|:---:|:---:|:---:|:---:|
| **Farm Fields / Light Forest** | **87.54%** | 74.05% | **66.99%** | 30,404,456 |
| **Buildings / Dry Land** | 47.52% | 53.50% | 33.63% | 460,844 |
| **Dense Forest** | 69.35% | **80.74%** | 59.50% | 15,592,280 |
| **Water Bodies** | **74.34%** | 71.98% | 57.66% | 2,627,343 |
| **Roads / Highways** | 18.66% | 62.47% | 16.78% | 915,077 |

- **Streamlined Single-Upload Inference**: Clean, distraction-free satellite image segmentation focused strictly on high-performance classification.
- **Sub-Second Speed**: Sliced $500 \times 500$ PNG tiles process in **under 0.2 seconds** with zero loading lag.

---

## 🎨 Semantic Land-Cover Classes

Roads, highways, buildings, and dry land are unified under the **Yellow** class, eliminating confusing black masks.

| ID | Class Name | Color Code | Hex | Description |
|:---:|:---|:---:|:---:|:---|
| 1 | **Dense Forest** | Deep Green | `#006400` | Thick canopy trees, dense woodlands, rough canopy texture |
| 2 | **Farm Fields / Pastures** | Light Green | `#22C55E` | Agricultural cropland, pastures, grass, mild smooth texture |
| 3 | **Water Bodies** | Blue | `#1E90FF` | Rivers, lakes, quarry pits, canals, specular smooth surface |
| 4 / 5 | **Buildings, Roads & Dry Land** | Yellow | `#FFD700` | Concrete roofs, residential/industrial buildings, roads, highways, asphalt, and dry bare soil |
| 6 | **Unclassified** | Grey | `#808080` | Cloud shadows, ambiguous transitional boundaries |

---

## 🚀 Quick Start

### 1. Installation
Ensure [uv](https://github.com/astral-sh/uv) or Python 3.12+ is installed:
```bash
# Clone the repository
git clone https://github.com/your-username/landclassifier.git
cd landclassifier

# Install dependencies with uv
uv sync
```

### 2. Run the Streamlit Web Application
```bash
uv run streamlit run frontend/app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the FastAPI Backend (Optional REST API)
```bash
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Benchmark Metrics: `GET http://127.0.0.1:8000/metrics`
- Classification: `POST http://127.0.0.1:8000/classify`

---

## 📊 Dataset Structure

```
data/
├── images/           # 11,366 500x500 PNG satellite orthophoto tiles
├── masks/            # 11,366 500x500 PNG ground-truth raster masks (classes 0..4)
├── train.txt         # 9,092 tile IDs (80% training set)
└── test.txt          # 2,274 tile IDs (20% test evaluation set)
```

To re-tile orthophotos at custom resolutions or split ratios:
```bash
uv run python data/split.py
```

---

## 🧪 Testing

Run the automated test suite with pytest:
```bash
uv run pytest
```
All unit tests verify image ingestion, edge-preserving bilateral filtering, 8-feature extraction, Random Forest inference, accuracy calculation, and REST endpoints.
