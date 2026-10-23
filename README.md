# Land Classification Using Traditional Image Processing

A land-cover classification system that processes satellite imagery using OpenCV, K-means clustering, and traditional image-processing techniques. The application uses FastAPI as the backend and Streamlit as the frontend to generate segmented land-cover maps, estimate class-wise area percentages, and evaluate classification performance.

## 1. Project Overview

### Objective

To develop an image-processing system that analyses satellite images, segments pixels according to their visual characteristics, and generates a colour-coded map representing different land-cover categories.

The project focuses on traditional image processing rather than deep learning.

### Problem Statement

Satellite images contain multiple land-cover types, such as vegetation, water bodies, built-up areas, and bare soil. Identifying these regions manually can be time-consuming.

This project aims to automate the initial segmentation and classification process using colour features, K-means clustering, and image-processing techniques.

### Proposed Solution

The system accepts a satellite image, preprocesses it, groups similar pixels using K-means clustering, assigns tentative land-cover labels, refines the resulting masks, and presents the results through an interactive web interface.

## 2. Project Objectives

1. Preprocess satellite images using resizing, noise reduction, and colour-space conversion.
2. Segment images into visually similar regions using K-means clustering.
3. Identify tentative land-cover categories using colour and optional texture features.
4. Refine segmentation masks using morphological operations.
5. Estimate the percentage of image pixels belonging to each category.
6. Visualize the original image, segmentation map, and classified output.
7. Evaluate performance using manually labelled ground-truth masks.
8. Provide an interactive interface for uploading images and downloading results.

## 3. Land-Cover Categories

The initial system will consider five categories:

| Category | Description | Display colour |
|---|---|---|
| Vegetation | Crops, grass, trees and green areas | Green |
| Water bodies | Rivers, ponds and lakes | Blue |
| Built-up areas | Buildings, roads and concrete surfaces | Red |
| Bare soil | Exposed earth and dry soil | Yellow |
| Other / uncertain | Mixed or ambiguous regions | Grey |

**Important:** These categories are approximate. K-means identifies clusters of similar pixels; it does not inherently understand land-cover semantics. A separate labelling step is required, and some classes may be difficult to distinguish using RGB imagery alone.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend API | FastAPI |
| API server | Uvicorn |
| Data validation | Pydantic |
| Image processing | OpenCV |
| Numerical computation | NumPy |
| Clustering | scikit-learn |
| Optional texture analysis | scikit-image |
| Visualization | Matplotlib |
| Data analysis | pandas |
| Frontend | Streamlit |
| Testing | pytest, HTTPX |
| Environment and dependencies | uv |
| Version control | Git and GitHub |

No GPU or deep-learning framework is required for the initial implementation.

## 5. System Architecture

The project follows a modular monolithic architecture. The processing modules, API routes, and application services run within one FastAPI backend.

```text
                  USER
                   |
                   v
          +------------------+
          | Streamlit UI     |
          | Image Upload     |
          | Settings         |
          | Result Display   |
          +------------------+
                   |
              HTTP Request
                   |
                   v
          +------------------+
          | FastAPI Backend  |
          | Validation       |
          | API Endpoints    |
          +------------------+
                   |
                   v
          +------------------+
          | Processing       |
          | Pipeline         |
          +------------------+
                   |
          +--------+---------+
          |        |         |
          v        v         v
      OpenCV    K-means   Label Mapping
          |        |         |
          +--------+---------+
                   |
                   v
          +------------------+
          | Mask Refinement  |
          | Area Statistics  |
          | Evaluation       |
          +------------------+
                   |
                   v
          +------------------+
          | Results          |
          | Classified Image |
          | JSON Statistics  |
          +------------------+
                   |
                   v
             Streamlit UI
```

### Component responsibilities

**Streamlit frontend**
- Upload satellite images.
- Configure segmentation parameters.
- Display original and processed images.
- Show land-cover statistics and charts.
- Download the classified image and CSV report.

**FastAPI backend**
- Accept and validate image uploads.
- Expose processing endpoints.
- Invoke the image-processing pipeline.
- Return result images and structured statistics.
- Handle invalid inputs and processing errors.

**Image-processing engine**
- Preprocess images.
- Extract colour and optional texture features.
- Perform K-means clustering.
- Assign tentative land-cover labels.
- Refine segmentation masks.
- Calculate class statistics and evaluation metrics.

## 6. Image-Processing Methodology

### Stage 1: Image Acquisition

Obtain satellite or aerial images from public datasets or satellite imagery services.

Initial dataset options:

- EuroSAT: https://github.com/phelber/EuroSAT
- Sentinel-2 imagery: https://dataspace.copernicus.eu/
- UC Merced Land Use Dataset: http://weegee.vision.ucmerced.edu/datasets/landuse.html

EuroSAT is convenient for initial experiments. However, its standard image-level labels are not pixel-level ground truth. Manual masks or a suitable pixel-labelled dataset will be needed for pixel-level evaluation.

### Stage 2: Preprocessing

1. Load the image using OpenCV.
2. Validate the image and its dimensions.
3. Resize to a consistent working resolution.
4. Apply optional Gaussian blurring to reduce noise.
5. Convert BGR imagery into CIELAB colour space.

### Stage 3: Feature Extraction

Initially, use the three CIELAB channels as pixel features.

Optionally, extract texture features using the Gray-Level Co-occurrence Matrix (GLCM) to investigate whether texture helps distinguish visually similar regions.

### Stage 4: K-means Segmentation

Represent each pixel as a feature vector and cluster the pixels using K-means.

Experiment with:

- K = 3
- K = 5
- K = 7

Compare the segmentation maps and quantitative metrics for each configuration.

K-means minimizes within-cluster squared distances. Cluster identifiers are arbitrary and must not be treated as fixed land-cover labels.

### Stage 5: Land-Cover Labelling

Assign tentative semantic labels to clusters using interpretable rules based on cluster colours and optional texture features.

For example, a cluster with appropriate vegetation-like colour characteristics may be assigned to vegetation.

Inspect representative regions before accepting a mapping. Include an uncertain category when the available features do not support a reliable label.

### Stage 6: Mask Refinement

Use morphological operations to improve the masks:

- Opening to reduce small foreground noise.
- Closing to fill small gaps.
- Optional connected-component filtering to remove isolated regions.

Compare the results before and after refinement.

### Stage 7: Area Estimation

Calculate the number of pixels assigned to each class and their proportion of the image.

For class \(c\):

Area percentage(c) = (pixels belonging to class c / total image pixels) × 100

These values represent image-area proportions. They are not physical land-area measurements unless the imagery's spatial resolution and georeferencing are known and handled correctly.

### Stage 8: Visualization

Generate:

- Original satellite image.
- K-means segmentation map.
- Labelled land-cover map.
- Class-wise area chart.
- Optional before-and-after refinement comparison.

## 7. Project Folder Structure

```text
land-classification/
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/
│       │   └── routes.py
│       ├── schemas/
│       │   └── classification.py
│       ├── services/
│       │   └── classifier.py
│       ├── processing/
│       │   ├── preprocessing.py
│       │   ├── segmentation.py
│       │   ├── labeling.py
│       │   ├── refinement.py
│       │   └── metrics.py
│       └── core/
│           └── config.py
├── frontend/
│   └── app.py
├── tests/
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

## 8. Micro-Task Development Roadmap

Development is divided into five phases and 24 micro-tasks. Each task should be implemented and tested before moving to the next.

### Phase A — Project Foundation

**A1. Repository and folder structure**
- Create the project repository.
- Create the folder structure.
- Initialize Git.
- Add the initial README.

Acceptance criteria: the repository exists, the folders are present, and the initial documentation is committed.

**A2. Python environment**
- Initialize the project using uv.
- Configure the Python version.
- Create the virtual environment.
- Generate the dependency lockfile.

Acceptance criteria: the environment can be recreated and dependencies can be installed consistently.

**A3. Dependencies and configuration**
- Install FastAPI, Uvicorn, OpenCV, NumPy, scikit-learn, Streamlit, Matplotlib, pandas, pytest, and HTTPX.
- Add scikit-image when texture analysis is introduced.
- Configure application settings.

Acceptance criteria: imports work and the development environment starts successfully.

**A4. Image loading and validation**
- Load a test satellite image.
- Check invalid file paths and unsupported images.
- Validate image dimensions.
- Handle grayscale images and supported image formats.

Acceptance criteria: valid images load correctly, and invalid inputs produce understandable errors.

### Phase B — Image-Processing Engine

**B1. Image preprocessing**
- Implement resizing.
- Add optional Gaussian blur.
- Convert BGR to CIELAB.
- Validate output dimensions and data types.

Acceptance criteria: preprocessing produces valid, consistent image arrays.

**B2. K-means segmentation**
- Reshape pixels into feature vectors.
- Implement configurable K-means clustering.
- Return cluster labels and cluster centres.
- Make experiments reproducible with a fixed random seed.

Acceptance criteria: clustering produces the expected number of clusters and a label for every pixel.

**B3. Cluster visualization**
- Map cluster identifiers to distinct display colours.
- Generate a segmentation image.
- Save and display the result.

Acceptance criteria: the segmentation map has the correct dimensions and visibly distinguishes clusters.

**B4. Cluster statistics**
- Count pixels in every cluster.
- Calculate cluster percentages.
- Verify that percentages sum to approximately 100%.

Acceptance criteria: cluster statistics are internally consistent.

**B5. Land-cover labelling**
- Implement interpretable cluster-to-class mapping rules.
- Add an uncertain class.
- Document limitations and ambiguous cases.

Acceptance criteria: predicted labels are inspectable and do not assume fixed K-means cluster identifiers.

**B6. Mask refinement**
- Implement morphological opening and closing.
- Configure the structuring element.
- Compare masks before and after refinement.

Acceptance criteria: refinement works without changing image dimensions or introducing invalid mask values.

### Phase C — FastAPI Backend

**C1. Application setup**
- Create the FastAPI application.
- Add `GET /health`.
- Run the application with Uvicorn.

Acceptance criteria: the application starts and the health endpoint responds successfully.

**C2. Image upload endpoint**
- Implement `POST /classify`.
- Accept multipart image uploads.
- Validate image size, dimensions and decoding.
- Reject invalid inputs safely.

Acceptance criteria: valid images are accepted and invalid uploads return appropriate HTTP errors.

**C3. Processing integration**
- Connect the API to the processing pipeline.
- Pass validated images and configuration parameters.
- Keep processing logic separate from route handlers.

Acceptance criteria: uploaded images are processed using the same tested pipeline.

**C4. Classification results**
- Return structured class statistics.
- Return cluster and label information.
- Provide access to the generated result image.

Acceptance criteria: the response follows a documented, consistent API contract.

**C5. Error handling and tests**
- Add API and processing tests.
- Test invalid uploads and invalid parameters.
- Test successful classification.
- Test response schemas.

Acceptance criteria: pytest passes for the implemented test cases.

### Phase D — Streamlit Frontend

**D1. Application shell**
- Create the Streamlit page.
- Add a title, description and sidebar.
- Organize the interface into clear sections.

Acceptance criteria: the interface starts and renders without errors.

**D2. Upload and preview**
- Add image upload.
- Display the original image.
- Show image dimensions and basic metadata.

Acceptance criteria: supported uploads are previewed correctly.

**D3. FastAPI integration**
- Send the image to the backend.
- Handle loading and error states.
- Display a useful message if the backend is unavailable.

Acceptance criteria: the frontend successfully communicates with the backend.

**D4. Result visualization**
- Display the segmentation and classified map.
- Provide an original-versus-result comparison.
- Show the chosen K value and relevant settings.

Acceptance criteria: results correspond to the image submitted by the user.

**D5. Statistics and downloads**
- Display class-wise area percentages.
- Create charts using Matplotlib.
- Allow downloads of result images and CSV statistics.

Acceptance criteria: exported files contain the displayed results and correct statistics.

### Phase E — Evaluation and Finalization

**E1. Ground-truth masks**
- Select representative test images.
- Manually label land-cover regions.
- Store ground-truth masks with consistent class identifiers.

Acceptance criteria: each evaluation image has a matching, aligned ground-truth mask.

**E2. Evaluation metrics**
- Implement pixel accuracy.
- Calculate per-class precision and recall.
- Calculate per-class IoU and mean IoU.
- Report class-wise F1 scores.

Acceptance criteria: metrics are computed against the correct ground-truth labels, with a documented mapping between predicted and true classes.

**E3. Parameter comparison**
- Evaluate K = 3, 5 and 7.
- Compare preprocessing configurations.
- Compare masks before and after refinement.
- Test optional texture features.

Acceptance criteria: results are recorded in a comparison table using the same held-out images.

**E4. End-to-end testing**
- Test upload, processing, visualization and downloads.
- Test invalid inputs and backend errors.
- Verify the application on several different images.

Acceptance criteria: the complete workflow functions without critical errors.

**E5. Report and demonstration**
- Document objectives and methodology.
- Include architecture and processing diagrams.
- Present results and evaluation metrics.
- Discuss limitations and possible improvements.

Acceptance criteria: the project can be demonstrated and explained reproducibly.

## 9. Evaluation Metrics

The project will use both unsupervised clustering metrics and supervised segmentation metrics.

| Metric | Purpose |
|---|---|
| Inertia | Measures within-cluster squared distances |
| Silhouette score | Measures cluster separation and cohesion |
| Pixel accuracy | Fraction of correctly classified pixels |
| Precision | Measures the proportion of predicted class pixels that are correct |
| Recall | Measures the proportion of true class pixels that are recovered |
| F1 score | Harmonic mean of precision and recall |
| Intersection over Union (IoU) | Measures overlap between predicted and true class regions |
| Mean IoU | Average IoU across the evaluated classes |
| Processing time | Measures the time required to process an image |

For class \(c\):

IoU(c) = TP(c) / [TP(c) + FP(c) + FN(c)]

Pixel accuracy and IoU require ground-truth masks. Predicted and reference labels must be aligned and mapped consistently before computing the metrics.

Inertia and silhouette score help evaluate clustering structure, but they do not prove that a cluster corresponds to the correct semantic land-cover category.

No accuracy target is assumed in advance. Performance will be reported after experiments.

## 10. Dataset and Experimental Plan

Start with a small, diverse set of satellite images containing vegetation, water, built-up areas and bare soil.

Suggested approach:

1. Use EuroSAT RGB images for initial experiments.
2. Select mixed-landscape imagery for the final demonstration.
3. Create manual pixel-level ground-truth masks for representative images.
4. Use separate development and evaluation images.
5. Compare K values and refinement settings on the same evaluation set.
6. Record processing time, segmentation metrics and qualitative observations.

Avoid treating EuroSAT's image-level class labels as pixel-level ground truth.

## 11. API Design

Initial endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend health |
| POST | `/classify` | Upload an image and perform segmentation and labelling |

The classification endpoint will accept an image and configurable parameters, including K, blur settings and refinement options.

The response will provide class statistics, configuration details, and the generated output image through a documented response format or a separate result endpoint if needed.

The exact response schema will be finalized during Phase C.

## 12. Security and Reliability

- Validate uploaded file types and decoded image contents.
- Enforce maximum upload sizes and image dimensions.
- Handle malformed files and processing errors.
- Avoid accepting arbitrary filesystem paths from users.
- Use controlled temporary-file handling where needed.
- Keep backend errors understandable without exposing internal details.
- Avoid storing uploaded images permanently unless explicitly required.

## 13. Expected Deliverables

1. Working Python image-processing pipeline.
2. FastAPI backend with documented endpoints.
3. Streamlit user interface.
4. Segmented and colour-coded land-cover maps.
5. Class-wise area statistics and downloadable CSV reports.
6. Ground-truth masks and quantitative evaluation results.
7. Automated tests.
8. Project report, screenshots and demonstration.

## 14. Limitations

- K-means is unsupervised and does not directly identify semantic land-cover categories.
- RGB colour alone cannot reliably distinguish all land types.
- Shadows, seasonal changes, image resolution and lighting can affect results.
- Morphological refinement can remove noise but may also remove small legitimate regions.
- Area percentages represent image pixels unless geographic calibration is performed.
- Results must be evaluated on representative images before making accuracy claims.

## 15. Future Improvements

Potential extensions include:

- Texture-based classification using GLCM features.
- Additional spectral bands and vegetation indices from multispectral imagery.
- Georeferenced area estimation.
- Comparison against Random Forest or other supervised methods.
- Multi-date satellite image comparison for land-cover change detection.
- Deployment of the FastAPI backend and Streamlit frontend.

## 16. Development Strategy

The project will be implemented incrementally, following the five development phases.

Each micro-task must have a clear output, acceptance criteria and appropriate tests. The image-processing engine will be developed and validated before frontend integration.

The initial priority is a reliable K-means segmentation pipeline. Semantic labelling, evaluation and user-interface features will be added only after the underlying processing stages work correctly.

**Project goal:** Build an explainable, reproducible land-cover segmentation system using traditional image processing, with a practical interface and measurable experimental results.
