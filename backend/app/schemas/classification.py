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
    algorithm: str = "random_forest"
    n_clusters: int = 5
    apply_blur: bool = True
    blur_kernel: int = 5
    use_bilateral: bool = True
    refine_mask: bool = True
    refine_kernel: int = 3

class EvaluationMetrics(BaseModel):
    has_ground_truth: bool = False
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mean_iou: Optional[float] = None
    per_class: Optional[Dict[str, Any]] = None

class ClassificationResponse(BaseModel):
    status: str = "success"
    algorithm: str = "random_forest"
    processing_time_ms: float
    original_dimensions: Dimensions
    processed_dimensions: Dimensions
    parameters: ClassificationParameters
    cluster_mapping: Dict[str, str] = Field(default_factory=dict)
    classes: List[ClassAreaStat]
    original_image_base64: str
    segmented_image_base64: str
    evaluation: Optional[EvaluationMetrics] = None

class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "Land Classification System"
    version: str = "1.0.0"
