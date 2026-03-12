


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.graph_objects as go
import os

# ── CONFIGURATION ──
CSV_PATH = "mnist_dataset (1).csv"  # Assumes file is in the same directory

st.set_page_config(page_title="Digit Classifier", page_icon="✏️", layout="wide")

# Professional Dark Theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0b0c15; color: #e2e8f0; }
    h1, h2, h3, p, div, span, label { color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    
    /* Clean Card Layout */
    .main-card {
        background-color: #151725; border: 1px solid #2d3748; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Status Indicator */
    .status-badge {
        display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-bottom: 20px;
        background-color: #2d3748; color: #a0aec0;
    }
    .status-badge.ready { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }

    /* Canvas Wrapper */
    .canvas-box { background: white; border-radius: 8px; overflow: hidden; display: flex; justify-content: center; align-items: center; }
    
    /* Prediction Display */
    .result-digit { 
        font-size: 5rem; font-weight: 800; line-height: 1; 
        background: linear-gradient(135deg, #6366f1, #a855f7); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%; border-radius: 8px; font-weight: 600; padding: 0.75rem; border: none; margin-top: 10px;
    }
    .btn-primary { background-color: #6366f1; color: white; }
    .btn-secondary { background-color: #2d3748; color: white; }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING & TRAINING (Backend Logic) ──
@st.cache_resource
def load_data_and_train():
    """Loads CSV and trains the model automatically on startup."""
    
    # 1. Check File
    if not os.path.exists(CSV_PATH):
        return None, f"Error: File '{CSV_PATH}' not found. Please place it in the app directory."

    with st.spinner("Loading dataset and training model... (This runs once)"):
        try:
            # Load Data
            df = pd.read_csv(CSV_PATH)
            
            # Preprocess (Assumes Col 0 is Label)
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values.astype('float32') / 255.0
            X = X.reshape(-1, 28, 28, 1)

            # Train/Val Split
            split = int(len(X) * 0.9) # Use 90% for train for speed
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # Model Architecture (Optimized for accuracy)
            model = models.Sequential([
                layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                layers.MaxPooling2D((2,2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Train
            model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=0)
            
            return model, "Ready"

        except Exception as e:
            return None, f"Error processing data: {str(e)}"

# Initialize Model
model, status_msg = load_data_and_train()

# ── UI LAYOUT ──
# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>✏️ Professional Digit Classifier</h1>", unsafe_allow_html=True)

# Status Bar
status_color = "ready" if model else ""
st.markdown(f"<div class='status-badge {status_color}'>📊 Status: {status_msg}</div>", unsafe_allow_html=True)

if not model:
    st.stop() # Stop if model failed to load

# Main Workspace
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 🎨 Draw Digit")
    
    from streamlit_drawable_canvas import st_canvas
    
    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=25,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas_main"
    )
    
    # Controls
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Canvas", use_container_width=True):
            st.rerun()
    with c2:
        predict_btn = st.button("Predict", use_container_width=True, type="primary")
        
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Prediction")
    
    if predict_btn and canvas_result.image_data is not None:
        # 1. Convert to Grayscale & Resize
        pil_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
        pil_img = pil_img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(pil_img)
        
        # 2. CRITICAL: Thresholding (Remove gray blur)
        # Makes edges sharp like MNIST data
        img_array = (img_array > 200).astype('float32') * 255
        
        # 3. Invert (White text on black background)
        img_array = 255.0 - img_array
        
        # 4. Normalize & Predict
        img_input = img_array.astype('float32') / 255.0
        img_input = img_input.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img_input, verbose=0)
        digit = int(np.argmax(prediction))
        conf = float(np.max(prediction)) * 100
        
        # Display Results
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 0.9rem; color: #a0aec0; margin-bottom: 5px;'>PREDICTED DIGIT</div>
            <div class='result-digit'>{digit}</div>
            <div style='font-size: 1.2rem; color: #a855f7; margin-top: 10px;'>{conf:.1f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Debug View (Optional but professional)
        with st.expander("Debug: What the model sees"):
            st.caption("If this 28x28 image doesn't look like your drawing, the prediction may be wrong.")
            st.image(img_array, width=100)
            
        # Chart
        probs = prediction[0]
        colors = ["#a855f7" if i == digit else "#2d3748" for i in range(10)]
        fig = go.Figure(data=[go.Bar(x=list(range(10)), y=probs, marker_color=colors)])
        fig.update_layout(
            plot_bgcolor="#151725", paper_bgcolor="#151725", margin=dict(l=0, r=0, t=20, b=0),
            font=dict(color="#a0aec0"), showlegend=False,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0,1])
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
    else:
        st.markdown("<div style='text-align:center; color: #4a5568; padding: 40px;'>Draw on the canvas and click Predict</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
