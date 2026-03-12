import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.graph_objects as go
import os, io

st.set_page_config(page_title="Digit Classifier", page_icon="✏️", layout="wide")

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
  .subtitle { text-align: center; color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem; }
  .card { background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
  .card-title { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #667eea; margin-bottom: 1rem; }
  .prediction-box { background: linear-gradient(135deg, #667eea22, #764ba222); border: 2px solid #667eea55; border-radius: 20px; padding: 2rem; text-align: center; }
  .digit-display { font-size: 5rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; }
  .confidence-text { font-size: 1.4rem; font-weight: 600; color: #a78bfa; margin-top: 0.5rem; }
  .stButton > button { background: #1e1e35 !important; color: #c4b5fd !important; border: 1px solid #4c4c7a !important; border-radius: 10px !important; font-weight: 500 !important; }
  .stButton > button:hover { background: #667eea22 !important; border-color: #667eea !important; }
  .stSlider > div > div > div { background: #667eea !important; }
  .tip-box { background: #0d1117; border-left: 3px solid #667eea; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.82rem; color: #64748b; }
  .train-box { background: #0d1117; border: 1px solid #2d2d4e; border-radius: 12px; padding: 1rem; font-size: 0.82rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "mnist_model.h5"
CSV_PATH   = "mnist_dataset.csv"

# ── Train from CSV ──
@st.cache_resource
def train_model_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    y = df["label"].values
    X = df.drop("label", axis=1).values.astype("float32") / 255.0
    X = X.reshape(-1, 28, 28, 1)

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=256,
              validation_data=(X_val, y_val), verbose=0)
    model.save(MODEL_PATH)
    _, acc = model.evaluate(X_val, y_val, verbose=0)
    return model, round(acc * 100, 2)

# ── Load saved model ──
@st.cache_resource
def load_saved_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH), None
    return None, None

# ── Session state ──
if "tool"       not in st.session_state: st.session_state.tool = "pencil"
if "canvas_key" not in st.session_state: st.session_state.canvas_key = "canvas_0"
if "model"      not in st.session_state: st.session_state.model = None
if "accuracy"   not in st.session_state: st.session_state.accuracy = None

# ── Header ──
st.markdown('<div class="main-title">✏️ Digit Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Train from CSV · Draw a digit · Get instant AI prediction</div>', unsafe_allow_html=True)

# ── Canvas import ──
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# ══════════════════════════════════════════════
# TRAINING SECTION (top)
# ══════════════════════════════════════════════
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🏋️ Model Training</div>', unsafe_allow_html=True)

tr1, tr2, tr3 = st.columns([2, 1.5, 1.5])

with tr1:
    uploaded_csv = st.file_uploader("Upload mnist_dataset.csv", type=["csv"], label_visibility="collapsed")
    st.caption("Upload `mnist_dataset.csv`  (label + 784 pixel columns)")

with tr2:
    use_saved = os.path.exists(MODEL_PATH)
    if use_saved and st.session_state.model is None:
        m, _ = load_saved_model()
        st.session_state.model = m

    if st.button("🚀 Train from CSV", use_container_width=True, disabled=(uploaded_csv is None)):
        with st.spinner("Training CNN on your CSV… ~30 sec"):
            # save uploaded file temporarily
            csv_bytes = uploaded_csv.read()
            with open(CSV_PATH, "wb") as f:
                f.write(csv_bytes)
            m, acc = train_model_from_csv(CSV_PATH)
            st.session_state.model = m
            st.session_state.accuracy = acc
        st.success(f"✅ Done! Validation accuracy: **{acc}%**")

with tr3:
    model_status = "✅ Model ready" if st.session_state.model else ("⚡ Load saved model" if use_saved else "⚠️ No model yet")
    acc_display  = f"{st.session_state.accuracy}%" if st.session_state.accuracy else ("saved" if use_saved else "—")
    st.markdown(f"""
    <div class="train-box">
      <b style="color:#94a3b8">Status:</b> {model_status}<br>
      <b style="color:#94a3b8">Accuracy:</b> {acc_display}<br>
      <b style="color:#94a3b8">CSV rows:</b> 35,000<br>
      <b style="color:#94a3b8">Epochs:</b> 5
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# DRAW + PREDICT (bottom)
# ══════════════════════════════════════════════
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎨 Drawing Canvas</div>', unsafe_allow_html=True)

    t1, t2, t3, _ = st.columns([1, 1, 1, 2])
    with t1:
        if st.button("✏️ Pencil", use_container_width=True):
            st.session_state.tool = "pencil"
    with t2:
        if st.button("⬜ Eraser", use_container_width=True):
            st.session_state.tool = "eraser"
    with t3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.canvas_key = f"canvas_{np.random.randint(99999)}"
            st.rerun()

    sc1, sc2 = st.columns(2)
    with sc1:
        stroke_width = st.slider("Stroke Width", 5, 40, 20)
    with sc2:
        stroke_color = st.color_picker("Color", "#000000")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    canvas_result = None
    if CANVAS_AVAILABLE:
        is_eraser = st.session_state.tool == "eraser"
        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=stroke_width * 2 if is_eraser else stroke_width,
            stroke_color="#FFFFFF" if is_eraser else stroke_color,
            background_color="#FFFFFF",
            height=280, width=280,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key,
            display_toolbar=False,
        )
    else:
        st.error("Run: pip install streamlit-drawable-canvas")

    st.markdown('<div class="tip-box">💡 Draw clearly in the center for best results. Train the model first using the CSV above.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # ── Prediction ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 AI Prediction</div>', unsafe_allow_html=True)

    prediction = confidence = probabilities = None
    model = st.session_state.model

    if CANVAS_AVAILABLE and canvas_result is not None and canvas_result.image_data is not None and model is not None:
        img_array = canvas_result.image_data
        pil_img = Image.fromarray(img_array.astype("uint8"), "RGBA").convert("L")
        gray = np.array(pil_img)

        if gray.min() < 200:
            resized = pil_img.resize((28, 28), Image.LANCZOS)
            img_norm = np.array(resized).astype("float32")
            img_inv = (255.0 - img_norm) / 255.0
            img_input = img_inv.reshape(1, 28, 28, 1)
            preds = model.predict(img_input, verbose=0)[0]
            prediction = int(np.argmax(preds))
            confidence = float(preds[prediction]) * 100
            probabilities = preds

    if prediction is not None:
        st.markdown(f"""
        <div class="prediction-box">
            <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem">PREDICTED DIGIT</div>
            <div class="digit-display">{prediction}</div>
            <div class="confidence-text">{confidence:.1f}%</div>
            <div style="color:#64748b;font-size:0.8rem">confidence</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        msg = "Train model first ↑" if model is None else "Draw a digit to predict"
        st.markdown(f"""
        <div class="prediction-box" style="opacity:0.5">
            <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem">PREDICTED DIGIT</div>
            <div style="font-size:4rem;color:#334155">?</div>
            <div style="color:#475569;font-size:0.85rem;margin-top:0.5rem">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Class Probabilities</div>', unsafe_allow_html=True)

    probs  = probabilities if probabilities is not None else np.zeros(10)
    colors = ["#f093fb" if (prediction is not None and i == prediction) else "#667eea" for i in range(10)]

    fig = go.Figure(go.Bar(
        x=[str(i) for i in range(10)], y=(probs * 100).tolist(),
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside", textfont=dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
        margin=dict(t=10, b=10, l=10, r=10), height=220,
        xaxis=dict(title="Digit", gridcolor="#1e1e35", tickfont=dict(color="#94a3b8")),
        yaxis=dict(title="Probability (%)", gridcolor="#1e1e35", tickfont=dict(color="#64748b"), range=[0, 115]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
