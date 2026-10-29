from pathlib import Path
import traceback
from backend.app.services.classifier import LandClassifierService

def main():
    img_dir = Path(r"c:\Users\rakes\OneDrive\Desktop\landcover.ai.v1\images")
    tifs = sorted(list(img_dir.glob("*.tif")))
    print(f"Found {len(tifs)} TIFF files to test.\n")

    errors = []
    successes = []

    for i, tif_path in enumerate(tifs):
        file_size_mb = tif_path.stat().st_size / (1024 * 1024)
        print(f"[{i+1}/{len(tifs)}] Testing {tif_path.name} ({file_size_mb:.1f} MB)...", end=" ", flush=True)
        try:
            raw_bytes = tif_path.read_bytes()
            res = LandClassifierService.classify(raw_bytes, n_clusters=5)
            orig_w = res["original_dimensions"]["width"]
            orig_h = res["original_dimensions"]["height"]
            t_ms = res["processing_time_ms"]
            non_zero = [c["class_name"] for c in res["classes"] if c["percentage"] > 0]
            successes.append((tif_path.name, (orig_w, orig_h), t_ms, non_zero))
            print(f"PASSED in {t_ms:.0f}ms ({orig_w}x{orig_h}px) -> {non_zero}")
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            errors.append((tif_path.name, str(e), traceback.format_exc()))
            print(f"FAILED: {err_msg}")

    print(f"\n=======================================================")
    print(f"Summary: {len(successes)} succeeded, {len(errors)} failed.")
    if errors:
        print("Detailed Error Logs:")
        for name, msg, tb in errors:
            print(f"=== {name} ===\n{tb}\n")

if __name__ == "__main__":
    main()
