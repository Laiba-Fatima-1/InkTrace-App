import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import time

st.set_page_config(
    page_title="InkTrace — Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model as keras_load
    return keras_load('digit_model.h5')

model = load_model()
CLASS_NAMES = [str(i) for i in range(10)]

# ---------------------------------------------------------------------------
# Custom CSS — "Notebook Paper" design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kalam:wght@400;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #FBF3E4;
    --paper-line: #E3D6BE;
    --margin-red: #A8394A;
    --ink: #262138;
    --plum: #7A2E3B;
    --gold: #D9A404;
    --sage: #3F6C51;
    --muted: #7A7364;
}

.stApp {
    background:
        repeating-linear-gradient(
            to bottom,
            transparent 0px,
            transparent 37px,
            var(--paper-line) 38px
        ),
        var(--paper);
    font-family: 'Inter', sans-serif;
    color: var(--ink);
    position: relative;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 90px;
    width: 2px; height: 100%;
    background: var(--margin-red);
    opacity: 0.35;
    z-index: 0;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2rem; max-width: 1100px; }

h1, h2, h3 { font-family: 'Kalam', cursive; color: var(--ink); }

.hero-wrap { text-align: center; padding: 1rem 0 0.5rem 0; animation: fadeIn 0.8s ease; }
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--plum); font-weight: 600;
}
.hero-title {
    font-family: 'Kalam', cursive; font-size: 3.4rem; font-weight: 700;
    color: var(--ink); margin: 0.3rem 0 0.5rem 0; line-height: 1.1;
}
.hero-title span { color: var(--sage); }
.hero-sub {
    font-size: 1.05rem; color: var(--muted); max-width: 600px;
    margin: 0 auto; line-height: 1.6;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes inkReveal {
    from { opacity: 0; transform: scale(0.85) rotate(-3deg); }
    to { opacity: 1; transform: scale(1) rotate(0deg); }
}

.card {
    background: #FFFDF8;
    border: 1px solid var(--paper-line);
    border-radius: 4px 16px 16px 4px;
    padding: 1.8rem 2rem;
    box-shadow: 3px 4px 0px 0px var(--paper-line), 0 6px 20px rgba(122,46,59,0.06);
    margin-bottom: 1.2rem;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--sage); font-weight: 600; margin-bottom: 0.8rem; display: block;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Kalam', cursive; font-size: 1.05rem;
    background: #F1E7D2; border-radius: 10px 10px 0 0; padding: 0.5rem 1.4rem;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    background: #FFFDF8 !important; color: var(--plum) !important; font-weight: 700;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--plum), #5E2430);
    color: #FBF3E4; border: none; border-radius: 10px;
    padding: 0.75rem 2.2rem; font-weight: 600; font-size: 1rem;
    font-family: 'Inter', sans-serif; width: 100%;
    box-shadow: 0 4px 16px rgba(122,46,59,0.3); transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px) rotate(-0.3deg);
    box-shadow: 0 8px 22px rgba(122,46,59,0.4);
}

.result-wrap { text-align: center; animation: inkReveal 0.5s ease; padding: 0.5rem 0; }
.pred-digit {
    font-family: 'Kalam', cursive; font-size: 6rem; font-weight: 700;
    color: var(--plum); line-height: 1; margin: 0.2rem 0;
}
.pred-conf {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem;
    color: var(--sage); font-weight: 600;
}
.bar-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.25rem 0; }
.bar-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; width: 20px; color: var(--muted); }
.bar-track { flex: 1; background: #F1E7D2; border-radius: 6px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--sage), var(--gold)); }
.bar-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; width: 42px; text-align: right; color: var(--muted); }

.disclaimer {
    font-size: 0.82rem; color: var(--muted); text-align: center;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px dashed var(--paper-line);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <span class="eyebrow">Machine Learning · Deep Learning CNN</span>
    <div class="hero-title">Every digit tells<br>a <span>story in ink.</span></div>
    <div class="hero-sub">Draw a digit (0-9) on the canvas or upload a photo of your handwriting —
    a convolutional neural network will read it instantly, trained to 99.3% accuracy.</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------
def preprocess_image(pil_img):
    """Convert a PIL image (any mode) to normalized 28x28 array matching MNIST format."""
    img = pil_img.convert('L')
    img = ImageOps.invert(img) if np.array(img).mean() > 127 else img
    img = img.resize((28, 28))
    arr = np.array(img).astype('float32') / 255.0
    return arr.reshape(1, 28, 28, 1), arr

def predict(arr):
    proba = model.predict(arr, verbose=0)[0]
    pred_class = int(np.argmax(proba))
    return pred_class, proba

def render_result(pred_class, proba, source_img=None):
    col_a, col_b = st.columns([1, 1.3], gap="large")
    with col_a:
        if source_img is not None:
            st.image(source_img, width=180, caption="Processed input (28×28)")
        st.markdown(f"""
        <div class="result-wrap">
            <span class="section-label">Prediction</span>
            <div class="pred-digit">{pred_class}</div>
            <div class="pred-conf">{proba[pred_class]*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown('<span class="section-label">Confidence Across All Digits</span>', unsafe_allow_html=True)
        bars_html = ""
        for i in range(10):
            pct = proba[i] * 100
            bars_html += f"""
            <div class="bar-row">
                <span class="bar-label">{i}</span>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%;"></div></div>
                <span class="bar-pct">{pct:.1f}%</span>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["✏️  Draw It", "📤  Upload a Photo"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Draw a single digit, centered</span>', unsafe_allow_html=True)
    from streamlit_drawable_canvas import st_canvas
    canvas_col, btn_col = st.columns([1, 1])
    with canvas_col:
        canvas_result = st_canvas(
            stroke_width=18,
            stroke_color="#FBF3E4",
            background_color="#262138",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    with btn_col:
        st.write("")
        st.write("")
        classify_drawing = st.button("Read My Handwriting", key="btn_draw")
        clear_note = st.caption("Tip: draw large and centered, like filling most of the box.")

    if classify_drawing:
        if canvas_result.image_data is not None and canvas_result.image_data[:, :, 3].sum() > 0:
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('RGB')
            arr, preview = preprocess_image(img)
            with st.spinner("Tracing the ink..."):
                time.sleep(0.4)
                pred_class, proba = predict(arr)
            render_result(pred_class, proba, preview)
        else:
            st.warning("Please draw a digit first!")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Upload a clear image of a handwritten digit</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    classify_upload = st.button("Read Uploaded Digit", key="btn_upload")

    if classify_upload:
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            arr, preview = preprocess_image(img)
            with st.spinner("Tracing the ink..."):
                time.sleep(0.4)
                pred_class, proba = predict(arr)
            render_result(pred_class, proba, preview)
        else:
            st.warning("Please upload an image first!")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    Currently trained on <b>digits 0–9</b> (MNIST). A full digits + letters (A–Z) version
    is in progress — this app will be updated once that model is ready.
</div>
""", unsafe_allow_html=True)
