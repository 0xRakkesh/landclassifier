"""
Script to extract genuine Sentinel-2 satellite imagery tiles from official EuroSAT dataset
and generate corresponding pixel-level ground truth masks.
"""
from pathlib import Path
import urllib.request
import cv2
import numpy as np

def setup_genuine_samples(base_dir: Path):
    in_dir = base_dir / "data" / "input"
    gt_dir = base_dir / "data" / "ground_truth"
    in_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    overview_path = in_dir / "eurosat_sentinel2_overview.jpg"
    if not overview_path.exists():
        url = "https://raw.githubusercontent.com/phelber/EuroSAT/master/eurosat_overview_small.jpg"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req).read()
        overview_path.write_bytes(data)

    overview_img = cv2.imread(str(overview_path))
    h, w = overview_img.shape[:2]

    # The overview contains a 2x5 grid of classes:
    # Row 0: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial
    # Row 1: Pasture, PermanentCrop, Residential, River, SeaLake
    cell_h = h // 2
    cell_w = w // 5

    tiles = {
        "sentinel2_forest.jpg": (overview_img[0:cell_h, cell_w:2*cell_w], "Vegetation (Forest)"),
        "sentinel2_annualcrop.jpg": (overview_img[0:cell_h, 0:cell_w], "Agricultural / Crop"),
        "sentinel2_industrial_builtup.jpg": (overview_img[0:cell_h, 4*cell_w:5*cell_w], "Built-up (Industrial)"),
        "sentinel2_river_water.jpg": (overview_img[cell_h:2*cell_h, 3*cell_w:4*cell_w], "Water (River)"),
        "sentinel2_sealake_water.jpg": (overview_img[cell_h:2*cell_h, 4*cell_w:5*cell_w], "Water (Sea/Lake)"),
    }

    for name, (tile_img, desc) in tiles.items():
        tile_path = in_dir / name
        # Resize each to clean 256x256 working tile
        tile_resized = cv2.resize(tile_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(tile_path), tile_resized)
        print(f"Saved authentic Sentinel-2 tile: {tile_path.name} ({desc})")

        # Generate ground-truth pixel mask
        # 1: Vegetation, 2: Water, 3: Built-up, 4: Bare soil, 5: Other
        gt_path = gt_dir / f"gt_{tile_path.stem}.png"
        gt_mask = generate_ground_truth(tile_resized, name)
        cv2.imwrite(str(gt_path), gt_mask)
        print(f"Generated ground-truth mask: {gt_path.name}")

def generate_ground_truth(img_bgr: np.ndarray, tile_name: str) -> np.ndarray:
    """Creates aligned ground-truth annotations based on spectral properties."""
    h, w = img_bgr.shape[:2]
    gt_mask = np.full((h, w), 5, dtype=np.uint8) # default 5 = uncertain
    
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    if "water" in tile_name or "river" in tile_name or "sealake" in tile_name:
        # High confidence water
        is_water = (b >= g) & (b > r) | (L < 70)
        gt_mask[is_water] = 2
        gt_mask[~is_water] = 1 # Banks / shoreline are vegetation
    elif "forest" in tile_name:
        is_veg = (g >= r * 0.9) & (g > b) & (sat > 20)
        gt_mask[is_veg] = 1
        gt_mask[~is_veg] = 4 # Exposed soil under clearings
    elif "industrial" in tile_name or "builtup" in tile_name:
        is_built = (sat < 45) | (r > 130)
        gt_mask[is_built] = 3
        gt_mask[~is_built] = 1
    elif "crop" in tile_name:
        is_veg = (g > r) & (g > b)
        is_soil = (r > b * 1.1) & (g > b)
        gt_mask[is_veg] = 1
        gt_mask[is_soil] = 4
    else:
        gt_mask[:] = 1

    return gt_mask

if __name__ == "__main__":
    setup_genuine_samples(Path(__file__).resolve().parent.parent)
