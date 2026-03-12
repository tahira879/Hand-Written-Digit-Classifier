import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import plotly.graph_objects as go
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0f0f1a; color: #e2e8f0; }

  .main-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem;
  }
  .subtitle {
    text-align: center; color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem;
  }

  .card {
    background: #1a1a2e; border: 1px solid #2d2d4e;
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #667eea; margin-bottom: 1rem;
  }

  .prediction-box {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border: 2px solid #667eea55; border-radius: 20px;
    padding: 2rem; text-align: center;
  }
  .digit-display {
    font-size: 5rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .confidence-text {
    font-size: 1.4rem; font-weight: 600; color: #a78bfa; margin-top: 0.5rem;
  }
  .confidence-label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }

  .tool-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 16px; border-radius: 10px; font-size: 0.85rem;
    font-weight: 500; cursor: pointer; border: none; transition: all 0.2s;
  }

  .stButton > button {
    background: #1e1e35 !important; color: #c4b5fd !important;
    border: 1px solid #4c4c7a !important; border-radius: 10px !important;
    font-weight: 500 !important; transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #667eea22 !important; border-color: #667eea !important;
    color: #a78bfa !important;
  }

  .active-tool button {
    background: #667eea33 !important; border-color: #667eea !important;
    color: #a78bfa !important;
  }

  .stSlider > div > div > div { background: #667eea !important; }
  .stSlider label { color: #94a3b8 !important; font-size: 0.85rem !important; }

  .tip-box {
    background: #0d1117; border-left: 3px solid #667eea;
    border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.82rem; color: #64748b;
  }

  div[data-testid="column"] { gap: 0 !important; }
  .canvas-wrapper canvas { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "mnist_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

# ─── Session State ───────────────────────────────────────────────────────────
if "tool" not in st.session_state:
    st.session_state.tool = "pencil"
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">✏️ Digit Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a digit (0–9) on the canvas — the AI will recognize it instantly</div>', unsafe_allow_html=True)

# ─── Try importing drawable canvas ───────────────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# ─── Layout ──────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎨 Drawing Canvas</div>', unsafe_allow_html=True)

    # ── Toolbar row ──
    t1, t2, t3, spacer = st.columns([1, 1, 1, 2])
    with t1:
        if st.button("✏️ Pencil", key="btn_pencil", use_container_width=True):
            st.session_state.tool = "pencil"
    with t2:
        if st.button("⬜ Eraser", key="btn_eraser", use_container_width=True):
            st.session_state.tool = "eraser"
    with t3:
        if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
            st.session_state.clear_canvas = True
            st.rerun()

    current_tool = st.session_state.tool

    # ── Stroke settings ──
    sc1, sc2 = st.columns(2)
    with sc1:
        stroke_width = st.slider("Stroke Width", 5, 40, 20, 1, key="stroke_width")
    with sc2:
        stroke_color = st.color_picker("Stroke Color", "#000000", key="stroke_color")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Canvas ──
    if CANVAS_AVAILABLE:
        drawing_mode = "freedraw" if current_tool == "pencil" else "freedraw"
        eraser_color = "#FFFFFF"
        active_color = eraser_color if current_tool == "eraser" else stroke_color
        active_width = stroke_width * 2 if current_tool == "eraser" else stroke_width

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=active_width,
            stroke_color=active_color,
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode=drawing_mode,
            key="canvas_main" if not st.session_state.clear_canvas else f"canvas_{np.random.randint(9999)}",
            display_toolbar=False,
        )
        st.session_state.clear_canvas = False
    else:
        st.warning("⚠️ `streamlit-drawable-canvas` not installed. Run: `pip install streamlit-drawable-canvas`")
        canvas_result = None

    st.markdown('<div class="tip-box">💡 Draw clearly in the center of the canvas for best results. Use the eraser to fix mistakes.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # ── Prediction Panel ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 AI Prediction</div>', unsafe_allow_html=True)

    prediction_placeholder = st.empty()
    chart_placeholder = st.empty()

    # ── Process Image ──
    prediction = None
    confidence = None
    probabilities = None

    if CANVAS_AVAILABLE and canvas_result is not None and canvas_result.image_data is not None:
        img_array = canvas_result.image_data

        # Check if canvas has been drawn on (not all white)
        if img_array is not None:
            pil_img = Image.fromarray(img_array.astype("uint8"), "RGBA")
            gray_img = pil_img.convert("L")
            gray_array = np.array(gray_img)

            # Only predict if something is drawn
            if gray_array.min() < 200:
                # Resize to 28x28
                resized = gray_img.resize((28, 28), Image.LANCZOS)
                img_norm = np.array(resized).astype("float32")

                # Invert: MNIST expects white digit on black bg
                img_inverted = 255.0 - img_norm
                img_normalized = img_inverted / 255.0
                img_input = img_normalized.reshape(1, 28, 28, 1)

                if model is not None:
                    preds = model.predict(img_input, verbose=0)[0]
                    prediction = int(np.argmax(preds))
                    confidence = float(preds[prediction]) * 100
                    probabilities = preds

    # ── Display Prediction ──
    if prediction is not None:
        with prediction_placeholder.container():
            st.markdown(f"""
            <div class="prediction-box">
                <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem">PREDICTED DIGIT</div>
                <div class="digit-display">{prediction}</div>
                <div class="confidence-text">{confidence:.1f}%</div>
                <div class="confidence-label">confidence</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with prediction_placeholder.container():
            st.markdown("""
            <div class="prediction-box" style="opacity:0.5">
                <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem">PREDICTED DIGIT</div>
                <div style="font-size:4rem;color:#334155">?</div>
                <div style="color:#475569;font-size:0.85rem;margin-top:0.5rem">Draw a digit to predict</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Probability Chart ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Class Probabilities</div>', unsafe_allow_html=True)

    if probabilities is not None:
        colors = ["#667eea" if i != prediction else "#f093fb" for i in range(10)]
        fig = go.Figure(go.Bar(
            x=[str(i) for i in range(10)],
            y=(probabilities * 100).tolist(),
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            margin=dict(t=10, b=10, l=10, r=10),
            height=220,
            xaxis=dict(
                title="Digit", gridcolor="#1e1e35",
                tickfont=dict(color="#94a3b8"),
            ),
            yaxis=dict(
                title="Probability (%)", gridcolor="#1e1e35",
                tickfont=dict(color="#64748b"), range=[0, 115],
            ),
            showlegend=False,
        )
        with chart_placeholder.container():
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        with chart_placeholder.container():
            # Empty chart placeholder
            fig = go.Figure(go.Bar(
                x=[str(i) for i in range(10)],
                y=[0]*10,
                marker_color="#1e1e35",
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                xaxis=dict(title="Digit", gridcolor="#1e1e35", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Probability (%)", gridcolor="#1e1e35", tickfont=dict(color="#64748b"), range=[0,100]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Info ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">ℹ️ Model Info</div>', unsafe_allow_html=True)
    status = "✅ Loaded" if model else "⚠️ Not Trained"
    ds_loaded = os.path.exists("dataset_X.npy")
    ds_size = np.load("dataset_X.npy").shape[0] if ds_loaded else 0
    st.markdown(f"""
    <div style="font-size:0.82rem;color:#64748b;line-height:1.8">
      <b style="color:#94a3b8">Architecture:</b> CNN (4× Conv2D + Dense)<br>
      <b style="color:#94a3b8">Dataset:</b> Synthetic MNIST-style ({ds_size} samples)<br>
      <b style="color:#94a3b8">Built-in:</b> ✅ No download required<br>
      <b style="color:#94a3b8">Input:</b> 28×28 grayscale<br>
      <b style="color:#94a3b8">Output:</b> 10 classes (0–9)<br>
      <b style="color:#94a3b8">Model:</b> {status}
    </div>
    """, unsafe_allow_html=True)
    if not model:
        st.warning("Run `generate_dataset.py` to train the model first.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Dataset Preview (bottom) ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📂 View Built-in Synthetic Dataset  (2 000 samples · 0 external downloads)"):
    if os.path.exists("dataset_X.npy") and os.path.exists("dataset_y.npy"):
        X_data = np.load("dataset_X.npy")
        y_data = np.load("dataset_y.npy")

        st.markdown(f"""
        <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1rem">
          <div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;padding:0.8rem 1.2rem;text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#a78bfa">{len(X_data)}</div>
            <div style="font-size:0.75rem;color:#64748b">Total Samples</div>
          </div>
          <div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;padding:0.8rem 1.2rem;text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#a78bfa">10</div>
            <div style="font-size:0.75rem;color:#64748b">Classes (0–9)</div>
          </div>
          <div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;padding:0.8rem 1.2rem;text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#a78bfa">28×28</div>
            <div style="font-size:0.75rem;color:#64748b">Image Size</div>
          </div>
          <div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;padding:0.8rem 1.2rem;text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#a78bfa">200</div>
            <div style="font-size:0.75rem;color:#64748b">Samples / Digit</div>
          </div>
          <div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;padding:0.8rem 1.2rem;text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#22c55e">✅</div>
            <div style="font-size:0.75rem;color:#64748b">No Download</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Sample images — 5 random examples per digit**")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(10, 5, figsize=(7, 14))
        fig.patch.set_facecolor("#0f0f1a")
        for digit in range(10):
            idxs = np.where(y_data == digit)[0]
            chosen = np.random.choice(idxs, 5, replace=False)
            for col, idx in enumerate(chosen):
                ax = axes[digit][col]
                ax.imshow(X_data[idx].reshape(28, 28), cmap="inferno", vmin=0, vmax=1)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(str(digit), fontsize=14, color="#a78bfa",
                                  fontweight="bold", labelpad=6, rotation=0, va="center")
        fig.suptitle("Synthetic Dataset — Built-in (no external source)",
                     color="#94a3b8", fontsize=10, y=1.01)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=False)
        plt.close()

        st.markdown("""
        <div style="font-size:0.8rem;color:#475569;margin-top:0.5rem">
        <b style="color:#94a3b8">How it's generated</b> — each digit is drawn using geometric primitives
        (Bresenham lines + parametric arcs), then augmented with random rotation (±25°),
        translation (±2 px), variable stroke width, Gaussian blur, and pixel noise.
        Re-run <code>generate_dataset.py</code> to regenerate with a different seed or more samples.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Dataset not found. Run `python generate_dataset.py` to generate it.")
