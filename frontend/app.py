"""
Satellite Land-Cover Classification System - Streamlit Web Application
Minimalist, high-performance UI dedicated strictly to classification.
"""
import io
import os
import base64
import streamlit as st
from PIL import Image

from backend.app.services.classifier import LandClassifierService

st.set_page_config(
    page_title="Satellite Land-Cover Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean aesthetics
st.markdown("""
<style>
    .legend-box {
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 8px;
        border-radius: 3px;
        vertical-align: middle;
    }
    .main-header {
        margin-bottom: 8px;
    }
    .badge-bar {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 20px;
        display: flex;
        gap: 24px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Satellite Land-Cover Classification")

# Minimalist model benchmark badge
bench = LandClassifierService.get_benchmark_metrics()
acc = bench.get("overall_accuracy", 75.63)
prec = bench.get("macro_precision", 59.48)
miou = bench.get("mean_iou", 46.91)

st.markdown(f"""
<div class="badge-bar">
    <span>🌲 <b>Model:</b> Supervised Random Forest</span>
    <span>🎯 <b>Test Accuracy:</b> <b style="color: #4ade80;">{acc:.1f}%</b></span>
    <span>🔍 <b>Macro Precision:</b> <b style="color: #60a5fa;">{prec:.1f}%</b></span>
    <span>📐 <b>mIoU:</b> <b style="color: #a78bfa;">{miou:.1f}%</b></span>
</div>
""", unsafe_allow_html=True)

# Sidebar Color Legend ONLY
st.sidebar.subheader("🎨 Land-Cover Classes")
classes_legend = [
    ("#006400", "Deep Green", "Dense Forest"),
    ("#22C55E", "Light Green", "Farm Fields & Pastures"),
    ("#1E90FF", "Blue", "Water Bodies"),
    ("#FFD700", "Yellow", "Buildings, Roads & Dry Land"),
    ("#808080", "Grey", "Shadows / Unclassified"),
]
for hex_color, color_name, symbolizes in classes_legend:
    st.sidebar.markdown(
        f'<div style="font-size: 14px; margin-bottom: 10px; display: flex; align-items: center;">'
        f'<span class="legend-box" style="background-color: {hex_color};"></span>'
        f'<b>{color_name}</b>&nbsp;→&nbsp;<span>{symbolizes}</span></div>',
        unsafe_allow_html=True,
    )

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a satellite image (PNG, JPG, TIFF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Upload a satellite orthophoto tile to classify its land cover.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

    # Preview and Classify
    classify_btn = st.button("🚀 Classify Land Cover", type="primary", use_container_width=True)

    if classify_btn:
        with st.spinner("Classifying land-cover pixels..."):
            result = LandClassifierService.classify(
                image_bytes=file_bytes,
                file_name=file_name,
                algorithm="random_forest",
                apply_blur=True,
                blur_kernel=5,
                use_bilateral=True,
                refine=True,
                refine_kernel=3,
            )

            orig_bytes = base64.b64decode(result["original_image_base64"])
            seg_bytes = base64.b64decode(result["segmented_image_base64"])
            orig_img = Image.open(io.BytesIO(orig_bytes))
            seg_img = Image.open(io.BytesIO(seg_bytes))

            # Visual Display Side by Side
            st.subheader("Classification Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_img, caption="Satellite Image", use_container_width=True)
            with col2:
                st.image(seg_img, caption=f"Classified Land Cover ({result['processing_time_ms']} ms)", use_container_width=True)

            # Coverage Distribution
            st.markdown("---")
            st.subheader("Land-Cover Distribution")
            area_rows = []
            for s in result["classes"]:
                area_rows.append({
                    "Class": s["class_name"],
                    "Coverage": f"{s['percentage']:.2f}%",
                    "Pixels": f"{s['pixel_count']:,}",
                })
            st.dataframe(area_rows, use_container_width=True)
    else:
        # Just show the uploaded image preview before clicking classify
        img_preview = Image.open(io.BytesIO(file_bytes))
        st.image(img_preview, caption=f"Uploaded: {file_name} ({img_preview.width}×{img_preview.height} px)", width=400)
