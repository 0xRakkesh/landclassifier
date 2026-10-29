from pydantic import BaseModel
from typing import Tuple, Dict

class Settings(BaseModel):
    PROJECT_NAME: str = "Land Classification System"
    VERSION: str = "2.2.0"
    DEFAULT_K: int = 5
    ALLOWED_K_VALUES: list[int] = [3, 5, 7]
    DEFAULT_BLUR_KERNEL: int = 3
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
    MAX_IMAGE_DIMENSION: int = 16384
    MAX_FILE_SIZE_MB: int = 100
    
    # Class definitions and visual color palette (RGB)
    # 1: Dense Forest (deep green: #006400)
    # 2: Farm Fields / Light Forest (green: #22C55E)
    # 3: Water Bodies (blue: #1E90FF)
    # 4: Roads / Highways (black: #141414)
    # 5: Buildings / Dry Land (yellow: #FFD700)
    # 6: Unclassified (grey: #808080)
    CLASSES: Dict[str, Dict] = {
        "Dense Forest": {
            "id": 1,
            "color_rgb": (0, 100, 0),        # Deep green
            "color_hex": "#006400",
            "description": "Dense forest and thick canopy vegetation",
        },
        "Farm Fields / Light Forest": {
            "id": 2,
            "color_rgb": (34, 197, 94),      # Green
            "color_hex": "#22C55E",
            "description": "Crops, agricultural fields, pastures, and light forest",
        },
        "Water Bodies": {
            "id": 3,
            "color_rgb": (30, 144, 255),     # Blue
            "color_hex": "#1E90FF",
            "description": "Rivers, lakes, reservoirs, ponds, oceans, and canals",
        },
        "Roads / Highways": {
            "id": 4,
            "color_rgb": (255, 215, 0),      # Yellow (same class/color as Buildings & Dry Land)
            "color_hex": "#FFD700",
            "description": "Roads, highways, asphalt runways, and paved paths (Yellow)",
        },
        "Buildings / Dry Land": {
            "id": 5,
            "color_rgb": (255, 215, 0),      # Yellow (Gold)
            "color_hex": "#FFD700",
            "description": "Buildings, residential/industrial roofs, concrete, dry bare soil, and sand",
        },
        "Unclassified": {
            "id": 6,
            "color_rgb": (128, 128, 128),    # Grey
            "color_hex": "#808080",
            "description": "Mixed terrain, shadows, or ambiguous regions",
        },
    }

settings = Settings()
