from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Dimensions(BaseModel):
    width: int
    height: int

class ClassAreaStat(BaseModel):
    class_id: int
    class_name: str
    color_hex: str
    color_rgb: List[int]
    pixel_count: int
    percentage: float

class ClassificationParameters(BaseModel):
    algorithm: str = "kmeans"
    n_clusters: int
    dbscan_eps: Optional[float] = None
    dbscan_min_samples: Optional[int] = None
    apply_blur: bool
    blur_kernel: int
    refine_mask: bool
    refine_kernel: int

class ClassificationResponse(BaseModel):
    status: str = "success"
    algorithm: str = "kmeans"
    processing_time_ms: float
    original_dimensions: Dimensions
    processed_dimensions: Dimensions
    parameters: ClassificationParameters
    cluster_mapping: Dict[str, str]
    classes: List[ClassAreaStat]
    original_image_base64: str
    segmented_image_base64: str

class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "Land Classification System"
    version: str = "1.0.0"
