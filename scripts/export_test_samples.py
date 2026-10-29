"""
Export representative 512x512 and 1024x1024 test images from the LandCover.ai dataset.
Converts large .tif satellite orthophotos into standard web-ready PNG and JPG images
covering:
1. River/Canal Water + Fields
2. Open Lake Water
3. Urban Buildings, Rooftops, and Roads
4. Dense Forest and Woodland
5. Agricultural Farm Fields
6. Highway / Transportation Corridor
"""
from pathlib import Path
import cv2
import numpy as np

DATASET_DIR = Path(r"c:\Users\rakes\OneDrive\Desktop\landcover.ai.v1")
OUTPUT_DIRS = [
    Path(r"C:\Users\rakes\OneDrive\Desktop\landclassifier\data\test_samples"),
    Path(r"C:\Users\rakes\OneDrive\Desktop\landclassifier\data\input"),
]

TILES_CONFIG = [
    {
        "filename": "landcover_river_canal_water.png",
        "source_img": "M-33-20-D-c-4-2.tif",
        "crop": (4500, 5012, 1500, 2012),  # y1, y2, x1, x2 (512x512)
        "description": "Inland river/canal water with surrounding fields and trees",
    },
    {
        "filename": "landcover_lake_open_water.png",
        "source_img": "M-34-32-B-a-4-3.tif",
        "crop": (1500, 2012, 4500, 5012),  # 512x512
        "description": "Deep open lake water with shoreline vegetation",
    },
    {
        "filename": "landcover_urban_buildings_roads.png",
        "source_img": "M-33-20-D-c-4-2.tif",
        "crop": (1000, 1512, 7000, 7512),  # 512x512
        "description": "Residential rooftops, building structures, roads, and dry land",
    },
    {
        "filename": "landcover_dense_forest_woodland.png",
        "source_img": "M-33-20-D-c-4-2.tif",
        "crop": (500, 1012, 5000, 5512),   # 512x512
        "description": "Dense woodland canopy, thick forest trees",
    },
    {
        "filename": "landcover_agricultural_farm_fields.png",
        "source_img": "M-33-20-D-c-4-2.tif",
        "crop": (3000, 3512, 1000, 1512),  # 512x512
        "description": "Active agricultural fields, crop parcels, and pastures",
    },
    {
        "filename": "landcover_highway_transport.png",
        "source_img": "M-33-20-D-c-4-2.tif",
        "crop": (7000, 7512, 7000, 7512),  # 512x512
        "description": "Asphalt highway road network and intersections",
    },
    {
        "filename": "landcover_mixed_landscape_1024.png",
        "source_img": "M-33-20-D-c-4-2.tif",
        "crop": (4200, 5224, 1200, 2224),  # 1024x1024
        "description": "Large mixed scene: river water, forest, agricultural land, and roads",
    },
]

def main():
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)

    loaded_images = {}

    print(f"Loading source imagery from: {DATASET_DIR}")
    for item in TILES_CONFIG:
        src_name = item["source_img"]
        src_path = DATASET_DIR / "images" / src_name

        if not src_path.exists():
            print(f"Warning: {src_path} not found. Skipping.")
            continue

        if src_name not in loaded_images:
            print(f"Reading source TIFF: {src_name}...")
            loaded_images[src_name] = cv2.imread(str(src_path))

        full_img = loaded_images[src_name]
        y1, y2, x1, x2 = item["crop"]
        tile = full_img[y1:y2, x1:x2]

        for out_dir in OUTPUT_DIRS:
            out_file = out_dir / item["filename"]
            cv2.imwrite(str(out_file), tile)
            print(f"  -> Saved: {out_file.name} ({tile.shape[1]}x{tile.shape[0]} px)")

    print("\nSuccessfully converted and exported LandCover.ai test images!")

if __name__ == "__main__":
    main()
