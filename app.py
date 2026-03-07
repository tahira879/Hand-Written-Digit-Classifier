import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# --- PAGE CONFIG ---
st.set_page_config(page_title="TensorFlow Digit Recognizer", layout="wide")

# --- CUSTOM CSS (Premium Dark Mode) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363d; }
    
    /* Neon Border for Canvas */
    .canvas-border {
        border: 2px solid #00F2FF;
        border-radius: 15px;
        box-shadow: 0px 0px 20px rgba(0, 242, 255, 0.2);
        padding: 10px;
        background-color: #1E1E1E;
        display: inline-block;
    }
    
    /* Recognition Button Styling */
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    try:
        return tf.keras.models.load_model('digit_model.h5')
    except:
        st.error("Model 'digit_model.h5' not found. Please run the training script first!")
        return None

model = load_my_model()

# --- SIDEBAR: TOOLKIT ---
with st.sidebar:
    st.title("Controls")
    
    # Brush Palette
    st.subheader("🎨 Palette")
    stroke_color = st.color_picker("Pick Pencil Color", "#00F2FF")
    
    # Tool Selection (Pencil vs Eraser)
    tool_mode = st.radio("Tool", ("Pencil", "Eraser"))
    stroke_width = st.slider("Brush Thickness", 5, 50, 20)
    
    # Logic: Eraser simply paints the background color
    final_stroke_color = "#1E1E1E" if tool_mode == "Eraser" else stroke_color

    st.markdown("---")
    if st.button("🗑️ Clear Canvas", use_container_width=True):
        st.rerun()

# --- MAIN LAYOUT ---
st.title("TensorFlow Digit Recognizer")

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.write("### Minimalist Canvas")
    
    # Visualizing the Canvas with the border
    st.markdown('<div class="canvas-border">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=final_stroke_color,
        background_color="#1E1E1E",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.write("### Live Prediction Model")
    
    # Process image only if user has drawn something
    if canvas_result.image_data is not None and np.any(canvas_result.image_data > 0):
        # 1. Image Preprocessing
        # Convert to Grayscale
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        # Resize to 28x28 (Matching MNIST/Kaggle size)
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        if model:
            # 2. Prediction
            prediction = model.predict(img_array)
            res = np.argmax(prediction)
            prob = np.max(prediction)

            # 3. Dynamic UI Update
            st.metric(label="Predicted Digit", value=res, delta=f"{prob*100:.2f}% Confidence")
            
            # Show Probability bars for all digits
            for i in range(10):
                p_val = float(prediction[0][i])
                # Highlight the top guess with the user's chosen color
                bar_color = stroke_color if i == res else "#333"
                st.write(f"Digit {i}")
                st.progress(p_val)
    else:
        st.info("Draw a digit on the canvas to see the AI analysis.")

# --- FOOTER ---
st.markdown("---")
st.caption("Custom Build | Powered by TensorFlow & Streamlit")
