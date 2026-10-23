from pydantic import BaseModel
from typing import Tuple, Dict

class Settings(BaseModel):
    PROJECT_NAME: str = "Land Classification System"
    VERSION: str = "1.0.0"
    DEFAULT_K: int = 5
    ALLOWED_K_VALUES: list[int] = [3, 5, 7]
    DEFAULT_BLUR_KERNEL: int = 3
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
    MAX_IMAGE_DIMENSION: int = 4096
    MAX_FILE_SIZE_MB: int = 25
    
    # Class definitions and visual color palette (RGB)
    CLASSES: Dict[str, Dict] = {
        "Vegetation": {"id": 1, "color_rgb": (34, 139, 34), "color_hex": "#228B22", "description": "Crops, grass, trees and green areas"},
        "Water": {"id": 2, "color_rgb": (30, 144, 255), "color_hex": "#1E90FF", "description": "Rivers, ponds and lakes"},
        "Built-up": {"id": 3, "color_rgb": (220, 20, 60), "color_hex": "#DC143C", "description": "Buildings, roads and concrete surfaces"},
        "Bare soil": {"id": 4, "color_rgb": (218, 165, 32), "color_hex": "#DAA520", "description": "Exposed earth and dry soil"},
        "Other / uncertain": {"id": 5, "color_rgb": (128, 128, 128), "color_hex": "#808080", "description": "Mixed or ambiguous regions"},
    }

settings = Settings()
