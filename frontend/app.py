"""
Streamlit Web Application for Land Classification.
Provides an interactive dashboard for satellite image upload, parameter tuning,
algorithm selection (K-Means vs DBSCAN), side-by-side segmentation visualization,
class area charts, and report downloads.
"""
import base64
import io
import time
import requests
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Direct processing service fallback
from backend.app.services.classifier import LandClassifierService

st.set_page_config(
    page_title="Satellite Land Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #228b22;
        margin-bottom: 10px;
    }
    .legend-box {
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 8px;
        border-radius: 3px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🌍 Satellite Land Cover Classification")
st.markdown("""
Automated segmentation and classification of satellite imagery using **CIELAB color-space features**, 
**K-Means clustering / DBSCAN density clustering**, and **morphological mask refinement**.
""")

# Sidebar Navigation & Settings
st.sidebar.header("⚙️ Algorithm & Parameters")
api_url = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")

# Check backend status
backend_healthy = False
try:
    health_resp = requests.get(f"{api_url}/health", timeout=1.0)
    if health_resp.status_code == 200:
        backend_healthy = True
        st.sidebar.success("Backend API Connected ✅")
except Exception:
    st.sidebar.info("Backend offline — using Direct Engine mode 🚀")

st.sidebar.markdown("---")
# Algorithm Selection in Sidebar
st.sidebar.subheader("🔬 Clustering Algorithm")
algo_choice = st.sidebar.radio(
    "Choose Approach:",
    options=["K-Means (Centroid-based)", "DBSCAN (Density-based)"],
    index=0,
    help="K-Means partitions into K spherical clusters. DBSCAN discovers arbitrary-shaped clusters and detects noise.",
)

algorithm_key = "kmeans" if "K-Means" in algo_choice else "dbscan"

n_clusters = 5
dbscan_eps = 5.0
dbscan_min_samples = 25

if algorithm_key == "kmeans":
    n_clusters = st.sidebar.select_slider(
        "Number of Clusters (K)",
        options=[3, 5, 7],
        value=5,
        help="Higher K captures finer spectral variations.",
    )
else:
    st.sidebar.markdown("*DBSCAN Parameters in CIELAB Space:*")
    dbscan_eps = st.sidebar.slider(
        "Epsilon (ε radius in CIELAB)",
        min_value=2.0,
        max_value=18.0,
        value=5.0,
        step=0.5,
        help="Maximum color distance to belong to the same cluster. Lower = more clusters.",
    )
    dbscan_min_samples = st.sidebar.slider(
        "Min Core Samples",
        min_value=10,
        max_value=60,
        value=25,
        step=5,
        help="Minimum number of pixels in neighborhood to form a dense core cluster.",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing & Morphology")
apply_blur = st.sidebar.checkbox("Apply Gaussian Blur (Noise Reduction)", value=True)
blur_kernel = 3
if apply_blur:
    blur_kernel = st.sidebar.selectbox("Blur Kernel Size", options=[3, 5, 7], index=0)

refine_mask = st.sidebar.checkbox("Apply Morphological Refinement", value=True)
refine_kernel = 3
if refine_mask:
    refine_kernel = st.sidebar.selectbox("Refinement Kernel Size", options=[3, 5, 7], index=0)

# Color Legend in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Land-Cover Classes")
classes_legend = [
    ("Vegetation", "#228B22", "Crops, forests, grasslands"),
    ("Water", "#1E90FF", "Rivers, lakes, water bodies"),
    ("Built-up", "#DC143C", "Roads, buildings, concrete"),
    ("Bare soil", "#DAA520", "Exposed dry earth, sand"),
    ("Other / uncertain", "#808080", "Mixed / ambiguous / DBSCAN noise"),
]
for name, color, desc in classes_legend:
    st.sidebar.markdown(
        f'<div style="font-size: 13px; margin-bottom: 4px;">'
        f'<span class="legend-box" style="background-color: {color};"></span>'
        f'<b>{name}</b>: {desc}</div>',
        unsafe_allow_html=True,
    )

# File Uploader
st.subheader("1. Upload Satellite Image")
uploaded_file = st.file_uploader(
    "Choose a satellite image file (PNG, JPG, JPEG, TIFF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(file_bytes))
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("File Name", uploaded_file.name)
    col_info2.metric("Original Dimensions", f"{image.width} × {image.height} px")
    col_info3.metric("Selected Algorithm", algo_choice)

    classify_btn = st.button("🚀 Run Classification", type="primary", use_container_width=True)

    if classify_btn:
        with st.spinner(f"Processing with {algo_choice}..."):
            result_data = None
            try:
                if backend_healthy:
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    params = {
                        "algorithm": algorithm_key,
                        "n_clusters": n_clusters,
                        "dbscan_eps": dbscan_eps,
                        "dbscan_min_samples": dbscan_min_samples,
                        "apply_blur": apply_blur,
                        "blur_kernel": blur_kernel,
                        "refine_mask": refine_mask,
                        "refine_kernel": refine_kernel,
                    }
                    resp = requests.post(f"{api_url}/classify", files=files, params=params, timeout=40)
                    if resp.status_code == 200:
                        result_data = resp.json()
                    else:
                        st.warning(f"Backend returned error ({resp.status_code}), executing with direct engine...")
                        result_data = LandClassifierService.classify(
                            image_bytes=file_bytes,
                            algorithm=algorithm_key,
                            n_clusters=n_clusters,
                            dbscan_eps=dbscan_eps,
                            dbscan_min_samples=dbscan_min_samples,
                            apply_blur=apply_blur,
                            blur_kernel=blur_kernel,
                            refine=refine_mask,
                            refine_kernel=refine_kernel,
                        )
                else:
                    # In-process execution
                    result_data = LandClassifierService.classify(
                        image_bytes=file_bytes,
                        algorithm=algorithm_key,
                        n_clusters=n_clusters,
                        dbscan_eps=dbscan_eps,
                        dbscan_min_samples=dbscan_min_samples,
                        apply_blur=apply_blur,
                        blur_kernel=blur_kernel,
                        refine=refine_mask,
                        refine_kernel=refine_kernel,
                    )
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")

            if result_data:
                st.success(
                    f"Classification complete in **{result_data['processing_time_ms']} ms** using **{result_data['algorithm'].upper()}**! ✨"
                )

                # Decode result images
                orig_bytes = base64.b64decode(result_data["original_image_base64"])
                seg_bytes = base64.b64decode(result_data["segmented_image_base64"])
                
                orig_img = Image.open(io.BytesIO(orig_bytes))
                seg_img = Image.open(io.BytesIO(seg_bytes))

                st.subheader("2. Visual Comparison")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.image(orig_img, caption="Preprocessed Satellite Image", use_container_width=True)
                with col_right:
                    st.image(
                        seg_img,
                        caption=f"Color-Coded Map ({result_data['algorithm'].upper()})",
                        use_container_width=True,
                    )

                st.subheader("3. Land-Cover Area Breakdown")
                stats = result_data["classes"]
                df_stats = pd.DataFrame(stats)
                
                dominant_class = max(stats, key=lambda x: x["percentage"])
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Dominant Land-Cover Class", dominant_class["class_name"], f"{dominant_class['percentage']}%")
                col_m2.metric("Execution Time", f"{result_data['processing_time_ms']} ms")

                # Charts
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                    names = [s["class_name"] for s in stats]
                    pcts = [s["percentage"] for s in stats]
                    colors = [s["color_hex"] for s in stats]
                    
                    bars = ax_bar.barh(names, pcts, color=colors, edgecolor="black", linewidth=0.8)
                    ax_bar.set_xlabel("Coverage Percentage (%)")
                    ax_bar.set_xlim(0, 100)
                    ax_bar.set_title(f"Area Percentage ({result_data['algorithm'].upper()})")
                    for bar in bars:
                        width = bar.get_width()
                        ax_bar.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center")
                    plt.tight_layout()
                    st.pyplot(fig_bar)

                with chart_col2:
                    fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                    non_zero = [s for s in stats if s["percentage"] > 0]
                    if non_zero:
                        pie_pcts = [s["percentage"] for s in non_zero]
                        pie_labels = [s["class_name"] for s in non_zero]
                        pie_colors = [s["color_hex"] for s in non_zero]
                        ax_pie.pie(
                            pie_pcts,
                            labels=pie_labels,
                            colors=pie_colors,
                            autopct="%1.1f%%",
                            startangle=140,
                            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                        )
                    ax_pie.axis("equal")
                    ax_pie.set_title(f"Class Distribution ({result_data['algorithm'].upper()})")
                    plt.tight_layout()
                    st.pyplot(fig_pie)

                # Export Options
                st.subheader("4. Export Results")
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.download_button(
                        label=f"📥 Download Classified Map ({result_data['algorithm'].upper()}) (PNG)",
                        data=seg_bytes,
                        file_name=f"{result_data['algorithm']}_classified_{uploaded_file.name}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                with d_col2:
                    csv_data = df_stats[["class_id", "class_name", "pixel_count", "percentage"]].to_csv(index=False)
                    st.download_button(
                        label="📊 Download Area Statistics (CSV)",
                        data=csv_data,
                        file_name=f"{result_data['algorithm']}_statistics_{uploaded_file.name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
else:
    st.info("👆 Upload a satellite image above to begin classification.")
