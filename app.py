import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Digit Classifier", layout="centered")

# --- CUSTOM DARK THEME CSS ---
st.markdown("""
    <style>
    /* Midnight Charcoal Background */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    
    /* Beautiful Heading Container */
    .header-box {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3b82f6;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(59, 130, 246, 0.3);
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    div.stButton > button:hover { background-color: #1d4ed8; }
    </style>
    """, unsafe_allow_html=True)

# --- UI LAYOUT ---
st.markdown("<div class='header-box'><h1>🎨 AI Digit Classifier</h1><p>Draw a number below</p></div>", unsafe_allow_html=True)

# --- CANVAS ---
# Tooling is built into the component for pen/eraser/color
col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# --- INFERENCE ---
with col2:
    st.write("### Controls")
    if st.button("Predict Digit"):
        if canvas_result.image_data is not None:
            # Pre-processing for MNIST models
            img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img).reshape(1, 28, 28) / 255.0
            
            # Placeholder for your model
            # model = tf.keras.models.load_model('mnist_model.h5')
            # pred = model.predict(img_array)
            # st.success(f"Detected: {np.argmax(pred)}")
            st.info("Model not loaded yet. Add your .h5 file!")

    if st.button("Clear Canvas"):
        st.rerun()
