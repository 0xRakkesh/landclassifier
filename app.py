"""
Hugging Face Spaces Entry Point for Satellite Land-Cover Classification.
Powered by Supervised Random Forest and Multi-Spectral Feature Engineering.
"""
import io
import base64
import numpy as np
import cv2
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
        return None, {}

    # Convert incoming PIL Image / array to PNG bytes
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

    # Coverage distribution dict for Gradio Label
    coverage = {s["class_name"]: round(s["percentage"] / 100.0, 4) for s in result["classes"]}

    return seg_image, coverage

# Apply ZeroGPU decorator if running on Hugging Face Spaces
if HAS_SPACES:
    predict_land_cover = spaces.GPU(_predict)
else:
    predict_land_cover = _predict

# Create Gradio Blocks Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🛰️ Satellite Land-Cover Classification System")
    gr.Markdown(
        "Upload any satellite or aerial orthophoto to classify its land cover into "
        "**Dense Forest (Deep Green)**, **Farm Fields (Light Green)**, **Water (Blue)**, and **Buildings/Roads (Yellow)**."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Satellite Image")
            btn = gr.Button("🚀 Classify Land Cover", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="pil", label="Classified Land-Cover Map")
            output_stats = gr.Label(num_top_classes=5, label="Land Cover Distribution")

    btn.click(
        fn=predict_land_cover,
        inputs=[input_img],
        outputs=[output_img, output_stats],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
