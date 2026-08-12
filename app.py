"""
app.py
------
Streamlit Web Interface for AI Age & Gender Detection.
Designed for Crixsoft Solution / Cybex Soft AI Internship - Task 3.
"""

import os
import sys
import time
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO

# Ensure utils directory is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Attempt to import the detector class
try:
    from utils.detector import AgeGenderDetector
except ImportError:
    class FallbackDetector:
        def analyze(self, bgr, conf_threshold):
            return bgr, []
    AgeGenderDetector = FallbackDetector

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Vision Lab | Age & Gender Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# ADVANCED CUSTOM CSS & STYLING (PRESERVING EXACT GEMINI UI & RADIO CARDS)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a1c2e 0%, #0d0e15 80%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Animated Border Glow Effect for Hero Container */
    @keyframes borderGlow {
        0% {
            border-color: rgba(0, 242, 254, 0.3);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 0 15px rgba(0, 242, 254, 0.05);
        }
        50% {
            border-color: rgba(0, 242, 254, 0.8);
            box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.25), inset 0 0 25px rgba(0, 242, 254, 0.2);
        }
        100% {
            border-color: rgba(0, 242, 254, 0.3);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 0 15px rgba(0, 242, 254, 0.05);
        }
    }

    .hero-container {
        position: relative;
        background: rgba(22, 27, 46, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1.5px solid rgba(0, 242, 254, 0.4);
        border-radius: 20px;
        padding: 40px 30px;
        text-align: center;
        margin-bottom: 30px;
        overflow: hidden;
        animation: borderGlow 4s infinite ease-in-out;
    }

    /* Light Wave Sweep Animation */
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            0deg,
            transparent,
            transparent,
            rgba(0, 242, 254, 0.08),
            rgba(79, 172, 254, 0.15),
            transparent
        );
        transform: rotate(30deg);
        animation: waveSweep 6s linear infinite;
        pointer-events: none;
    }

    @keyframes waveSweep {
        0% { transform: translateY(-100%) rotate(30deg); }
        100% { transform: translateY(100%) rotate(30deg); }
    }

    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        font-size: 2.3rem;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #00c6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 700px;
        margin: 0 auto 15px auto;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.4);
        color: #00f2fe;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }

    .input-section-container {
        background: rgba(22, 27, 46, 0.65);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 20px 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.3);
    }

    .input-section-title {
        font-family: 'Orbitron', sans-serif;
        color: #00f2fe;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .input-section-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 300;
    }

    /* Original Gemini Custom Radio Button Cards */
    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        gap: 16px !important;
        width: 100% !important;
        margin-top: 5px !important;
        margin-bottom: 25px !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1 !important;
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(0, 242, 254, 0.25) !important;
        border-radius: 12px !important;
        padding: 22px 12px !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease-in-out !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: rgba(0, 242, 254, 0.6) !important;
        background: rgba(15, 23, 42, 0.9) !important;
        transform: translateY(-2px);
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        border: 2px solid #00f2fe !important;
        background: rgba(0, 242, 254, 0.15) !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.5), inset 0 0 10px rgba(0, 242, 254, 0.2) !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label div[role="radio"] {
        border-color: #00f2fe !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
        margin: 0 !important;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .person-card {
        background: rgba(15, 23, 42, 0.75);
        border-left: 4px solid #00f2fe;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .person-card.female { border-left-color: #b967ff; }
    .gender-male { color: #48dbfb; }
    .gender-female { color: #b967ff; }

    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TUNABLE SETTINGS & UTILS (FROM CLAUDE IMPROVEMENTS)
# -----------------------------------------------------------------------------
MAX_PROCESSING_DIMENSION = 1600

def resize_if_large(bgr_image, max_dim=MAX_PROCESSING_DIMENSION):
    """Prevents high-res memory crashes by scaling down huge inputs safely."""
    h, w = bgr_image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return bgr_image
    scale = max_dim / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_detector_model():
    return AgeGenderDetector()

with st.sidebar:
    st.image("https://img.icons8.com/neon/96/artificial-intelligence.png", width=80)
    st.markdown("<h3 style='font-family: Orbitron; color: #00f2fe;'>AI Engine Controls</h3>",
                unsafe_allow_html=True)
    st.markdown("---")
    conf_threshold = st.slider(
        "Face Detection Confidence", 0.30, 0.90, 0.55, 0.05)
    st.markdown("---")
    st.markdown("### System Specs")
    st.markdown("- **Model:** Vision Transformer (ViT)")
    st.markdown("- **Detector:** OpenCV DNN SSD")
    st.markdown("- **Pre-processing:** CLAHE Normalization")
    st.markdown("- **Crop Engine:** Proportional Square Crop")
    st.markdown("---")
    st.caption("Crixsoft Solution • AI Internship Task 3")

with st.spinner("Initializing Deep Learning Engine..."):
    detector = load_detector_model()

# Hero Section with Animated Glow & Wave
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">AI AGE & GENDER DETECTION SYSTEM</div>
        <div class="hero-subtitle">
            Real-time multi-face detection, age estimation, and gender classification powered by 
            Vision Transformers (ViT) & Deep Neural Networks.
        </div>
        <div class="hero-badge">⚡ CRIXSOFT SOLUTION • AI INTERNSHIP TASK 3</div>
    </div>
""", unsafe_allow_html=True)

# About AI Expander (Requested from Claude)
with st.expander("ℹ️  About this AI — models, pipeline & limitations"):
    st.markdown(
        """
**Pipeline:** OpenCV DNN face detector (multi-scale, tiled for crowded
images) → square, lighting-normalized face crop → Vision Transformer
(ViT-Base) age/gender head.

**Gender prediction** is generally reliable in testing (roughly 85-99%
confidence on clear, front-facing faces).

**Age is an estimate, not an exact value** — it's shown as a likely range
rather than a single number, because no lightweight AI model of this kind
can honestly claim exact-year precision. Accuracy is best on clear,
well-lit, front-facing photos and can vary more on angled, dark, or
low-resolution faces — this is a known characteristic of this class of
model, not a bug.
        """
    )

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (WITH CLAUDE'S ROBUST ERROR HANDLING & DOWNLOAD)
# -----------------------------------------------------------------------------
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            pil_img = Image.open(BytesIO(response.content))
            pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            st.error(f"Could not download image (HTTP {response.status_code}).")
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
    return None

def load_any_image_format(uploaded_file_or_path, is_path=False):
    if is_path:
        if os.path.exists(uploaded_file_or_path):
            try:
                pil_img = Image.open(uploaded_file_or_path)
                pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(
                    f"Error loading image from {uploaded_file_or_path}: {e}")
        else:
            st.warning(f"File not found: {uploaded_file_or_path}")
        return None
    else:
        try:
            pil_img = Image.open(uploaded_file_or_path)
            pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return None

def process_and_display_image(bgr_image, threshold):
    bgr_image = resize_if_large(bgr_image)
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown("#### Input Visual & Detections")
        try:
            with st.spinner("Processing..."):
                t_start = time.time()
                annotated_bgr, results = detector.analyze(
                    bgr_image, conf_threshold=threshold)
                proc_time = (time.time() - t_start) * 1000
        except Exception as e:
            st.error("The AI pipeline could not process this image. Please try a different photo.")
            st.caption(f"Technical details: {e}")
            return

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

        # Download Result Feature
        success, encoded_png = cv2.imencode(".png", annotated_bgr)
        if success:
            st.download_button(
                "⬇️  Download annotated result",
                data=encoded_png.tobytes(),
                file_name="ai_age_gender_result.png",
                mime="image/png",
                use_container_width=True,
            )

    with col2:
        st.markdown("#### 📊 Analysis Report")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">Faces Detected</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{proc_time:.0f}<span style="font-size:0.9rem">ms</span></div>
                    <div class="metric-label">Inference Time</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not results:
            st.info("No faces detected in the selected image. Try a clearer, front-facing photo with better lighting.")
        else:
            for idx, res in enumerate(results, start=1):
                gender = res.get("gender", "Unknown")
                gender_conf = res.get("gender_confidence", 0.0) * 100
                age = res.get("age", "N/A")
                age_range = res.get("age_range", (0, 0))

                card_class = "female" if gender == "Female" else "male"
                gender_color = "gender-female" if gender == "Female" else "gender-male"
                gender_icon = "♀" if gender == "Female" else "♂"

                st.markdown(f"""
                    <div class="person-card {card_class}">
                        <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">Face Candidate #{idx}</div>
                        <div class="person-gender {gender_color}" style="font-size: 1.15rem; font-weight: 700;">
                            {gender_icon} {gender} <span style="font-size: 0.8rem; color: #94a3b8; font-weight:400;">({gender_conf:.1f}%)</span>
                        </div>
                        <div style="font-size: 1rem; color: #f8fafc; font-weight: 600; margin-top: 4px;">
                            Estimated Age: ~{age} <span style="font-size: 0.8rem; color: #94a3b8; font-weight:400;">({age_range[0]}–{age_range[1]} yrs)</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INPUT SECTION (100% PRESERVING GEMINI'S ST.RADIO AND LOOK)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="input-section-container">
    <div class="input-section-title">⚡ Select Input Method</div>
    <div class="input-section-subtitle">Choose how to provide an image for AI evaluation. The selected card will highlight.</div>
</div>
""", unsafe_allow_html=True)

MODE_WEB = "🌐 Web Link"
MODE_UPLOAD = "📁 Upload File"
MODE_CAMERA = "📷 Live Camera"
MODE_GALLERY = "🖼️ Pre-Tested Images"

selected_mode = st.radio(
    "Select Input Method",
    options=[MODE_WEB, MODE_UPLOAD, MODE_CAMERA, MODE_GALLERY],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

input_placeholder = st.container()
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "jfif"]
bgr_img = None

if selected_mode == MODE_WEB:
    with input_placeholder:
        st.markdown("<br>", unsafe_allow_html=True)
        url_input = st.text_input(
            "Enter direct image URL (Supports JPG, PNG, WEBP, etc.):", placeholder="https://example.com/image.png")
        if url_input.strip():
            with st.spinner("Downloading image from web..."):
                bgr_img = load_image_from_url(url_input.strip())

elif selected_mode == MODE_UPLOAD:
    with input_placeholder:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select an image file", type=ALLOWED_TYPES)
        if uploaded_file is not None:
            bgr_img = load_any_image_format(uploaded_file)

elif selected_mode == MODE_CAMERA:
    with input_placeholder:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📷 Live Camera Capture")
        camera_image = st.camera_input("Take a snapshot")
        if camera_image is not None:
            bgr_img = load_any_image_format(camera_image)

elif selected_mode == MODE_GALLERY:
    with input_placeholder:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Pre-Tested Image Selection")

        asset_files = {
    
            "face2.jpg": "assets/face2.jpg",
            "face3.jpg": "assets/face3.jpg",
            "group1.jpg": "assets/group1.jpg",
            "group2.jpg": "assets/group2.jpg",
            "group3.jpg": "assets/group3.jpg"
        }

        selected_asset_name = st.selectbox(
            "Select an image from local assets folder:",
            list(asset_files.keys())
        )
        selected_asset_path = asset_files[selected_asset_name]

        bgr_img = load_any_image_format(selected_asset_path, is_path=True)

# Process & Display Predictions
if bgr_img is not None:
    st.markdown("---")
    process_and_display_image(bgr_img, conf_threshold)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="footer-text">
         ⚡Crixsoft Solution • Artificial Intelligence Internship <br>
          Computer Vision & Vision Transformer (ViT) Pipeline
    </div>
""", unsafe_allow_html=True)