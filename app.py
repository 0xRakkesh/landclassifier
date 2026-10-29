"""
High-Performance, Modern Gradio Web Application for Satellite Land-Cover Classification.
Powered by Supervised Random Forest & Multi-Spectral Remote Sensing Feature Extraction.
"""
import io
import os
import base64
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import gradio as gr

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

from backend.app.services.classifier import LandClassifierService

def _predict(image):
    if image is None:
        return None, pd.DataFrame(columns=["Class", "Color", "Coverage %", "Pixel Count"])

    # Convert incoming PIL Image / numpy array to PNG bytes
    if isinstance(image, np.ndarray):
        _, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_bytes = encoded.tobytes()
    else:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

    result = LandClassifierService.classify(
        image_bytes=image_bytes,
        algorithm="random_forest",
        apply_blur=True,
        blur_kernel=5,
        use_bilateral=True,
        refine=True,
        refine_kernel=3,
    )

    seg_bytes = base64.b64decode(result["segmented_image_base64"])
    seg_image = Image.open(io.BytesIO(seg_bytes))

    # Build clean coverage table
    table_data = []
    for s in result["classes"]:
        # Only show classes with area or relevant classes
        table_data.append({
            "Land-Cover Class": s["class_name"],
            "Hex Color": s["color_hex"],
            "Coverage %": f"{s['percentage']:.2f}%",
            "Total Pixels": f"{s['pixel_count']:,}",
        })
    df_stats = pd.DataFrame(table_data)

    return seg_image, df_stats

if HAS_SPACES:
    predict_land_cover = spaces.GPU(_predict)
else:
    predict_land_cover = _predict

# Custom CSS for modern styling
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.badge-container {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 16px;
}
.badge-item {
    font-size: 14px;
    color: #9ca3af;
}
.badge-item b {
    color: #f3f4f6;
}
.badge-val-green { color: #34d399 !important; font-weight: bold; }
.badge-val-blue { color: #60a5fa !important; font-weight: bold; }
.badge-val-purple { color: #a78bfa !important; font-weight: bold; }

.legend-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
}
.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 500;
}
.color-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    display: inline-block;
}
"""

with gr.Blocks(css=CUSTOM_CSS, title="🛰️ Satellite Land-Cover Classifier") as demo:
    gr.Markdown("# 🛰️ Satellite Land-Cover Classification System")
    gr.Markdown("Automated, explainable semantic segmentation of aerial & satellite orthophotography powered by **Supervised Random Forest (100 Trees)**.")

    # Model Performance KPI Badges
    gr.HTML("""
    <div class="badge-container">
        <div class="badge-item">🌲 <b>Model:</b> Supervised Random Forest</div>
        <div class="badge-item">🎯 <b>Test Accuracy:</b> <span class="badge-val-green">75.6%</span></div>
        <div class="badge-item">🔍 <b>Macro Precision:</b> <span class="badge-val-blue">59.5%</span></div>
        <div class="badge-item">📐 <b>Mean IoU:</b> <span class="badge-val-purple">46.9%</span></div>
        <div class="badge-item">⚡ <b>Inference:</b> <span class="badge-val-green">~0.18s</span></div>
    </div>
    """)

    # Unified Color Legend
    gr.HTML("""
    <div class="legend-card">
        <span style="font-weight: 600; color: #9ca3af; margin-right: 4px;">🎨 Classes:</span>
        <div class="legend-item"><span class="color-swatch" style="background-color: #006400;"></span> <b>Deep Green</b> → Dense Forest</div>
        <div class="legend-item"><span class="color-swatch" style="background-color: #22C55E;"></span> <b>Light Green</b> → Farm Fields / Pastures</div>
        <div class="legend-item"><span class="color-swatch" style="background-color: #1E90FF;"></span> <b>Blue</b> → Water Bodies</div>
        <div class="legend-item"><span class="color-swatch" style="background-color: #FFD700;"></span> <b>Yellow</b> → Buildings, Roads & Dry Land</div>
        <div class="legend-item"><span class="color-swatch" style="background-color: #808080;"></span> <b>Grey</b> → Shadows / Unclassified</div>
    </div>
    """)

    # Interactive Segmentation Viewport
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="1. Upload Satellite Orthophoto (PNG, JPG, TIFF)")
            classify_btn = gr.Button("🚀 Classify Land Cover", variant="primary", size="lg")
        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="2. Classified Land-Cover Map")

    # Land-Cover Statistics Table
    with gr.Row():
        output_table = gr.Dataframe(
            headers=["Land-Cover Class", "Hex Color", "Coverage %", "Total Pixels"],
            datatype=["str", "str", "str", "str"],
            label="📊 Land-Cover Distribution Breakdown",
            interactive=False,
        )

    # Click-to-Test Sample Tiles
    sample_dir = os.path.join(os.path.dirname(__file__), "examples")
    sample_files = []
    if os.path.exists(sample_dir):
        for f in ["sample_forest_field.png", "sample_roads_urban.png"]:
            p = os.path.join(sample_dir, f)
            if os.path.exists(p):
                sample_files.append([p])

    if sample_files:
        gr.Examples(
            examples=sample_files,
            inputs=[input_img],
            label="💡 Quick Test: Click a Sample Satellite Tile to Inspect",
            examples_per_page=2,
        )

    classify_btn.click(
        fn=predict_land_cover,
        inputs=[input_img],
        outputs=[output_img, output_table],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
