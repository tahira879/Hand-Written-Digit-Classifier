import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.graph_objects as go
import os

# ── PAGE CONFIG & THEME ──
st.set_page_config(page_title="Digit Classifier", page_icon="✏️", layout="wide")

# Custom CSS for Dark Theme & White Text
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0f0f1a;
        color: #ffffff;
    }
    
    /* General Text Overrides */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #ffffff !important;
    }
    
    /* Card Container Style */
    .card {
        background-color: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Step Badge */
    .step-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #764ba2;
        transform: translateY(-1px);
    }
    
    /* File Uploader Text */
    [data-testid="stFileUploader"] {
        color: white !important;
    }
    [data-testid="stFileUploader"] label {
        color: #b0b0d0 !important;
    }

    /* Slider Color */
    .stSlider > div > div > div {
        background-color: #667eea !important;
    }

    /* Canvas Container */
    .canvas-container {
        border: 2px solid #2d2d4e;
        border-radius: 12px;
        background: white;
        overflow: hidden;
    }
    
    /* Prediction Box */
    .pred-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border: 2px solid #667eea;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .digit {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0.0

# ── HEADER ──
st.markdown("<h1 style='text-align: center;'>✏️ AI Digit Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0b0d0;'>Train a model on your CSV • Draw a digit • See the magic</p>", unsafe_allow_html=True)

# ── STEP 1: TRAINING SECTION ──
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<span class='step-badge'>STEP 1</span> <h3 style='display:inline;'>Train the Model</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload `mnist_dataset (1).csv`", type=["csv"])
    
    if uploaded_file is not None:
        # Read CSV to check structure (assume label is first column)
        df = pd.read_csv(uploaded_file)
        st.write(f"📊 **Dataset Loaded:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        if st.button("🚀 Start Training", use_container_width=True):
            with st.spinner("Training model... This may take a minute."):
                # 1. Preprocess Data
                # Assumes first column is label, rest are pixels
                y = df.iloc[:, 0].values
                X = df.iloc[:, 1:].values.astype('float32') / 255.0
                X = X.reshape(-1, 28, 28, 1)

                # 2. Split Data (Simple 80/20 split)
                split = int(len(X) * 0.8)
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]

                # 3. Define CNN Model
                model = models.Sequential([
                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Flatten(),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(10, activation='softmax')
                ])

                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                # 4. Train
                # Using a small number of epochs for speed in the demo
                history = model.fit(X_train, y_train, epochs=5, 
                                    validation_data=(X_val, y_val), verbose=0)
                
                # 5. Save to Session State
                st.session_state.model = model
                st.session_state.model_trained = True
                val_acc = history.history['val_accuracy'][-1]
                st.session_state.accuracy = f"{val_acc*100:.2f}%"
                
                st.success(f"Training Complete! Validation Accuracy: {st.session_state.accuracy}")
                st.rerun()

with col2:
    if st.session_state.model_trained:
        st.info("✅ **Model Ready!** \n\nProceed to Step 2 to draw and predict.")
    else:
        st.warning("⚠️ **Waiting for model...** \n\nPlease upload a CSV file and click Train.")

st.markdown("</div>", unsafe_allow_html=True)

# ── STEP 2: DRAW & PREDICT SECTION ──
# Only show if model is trained
if st.session_state.model_trained:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-badge'>STEP 2</span> <h3 style='display:inline;'>Draw & Predict</h3>", unsafe_allow_html=True)

    # Import canvas component
    from streamlit_drawable_canvas import st_canvas

    col_draw, col_res = st.columns([1, 1])

    with col_draw:
        st.markdown("**Canvas**")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#000000", # Black ink
            background_color="#FFFFFF", # White paper
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if st.button("🔮 Predict Digit", use_container_width=True):
            if canvas_result.image_data is not None:
                # Process Image
                img_data = canvas_result.image_data
                # Convert RGBA to Grayscale PIL Image
                pil_img = Image.fromarray(img_data.astype('uint8'), 'RGBA').convert('L')
                
                # Resize to 28x28
                pil_img = pil_img.resize((28, 28), Image.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(pil_img)
                
                # Invert colors (MNIST is white on black, canvas is black on white)
                img_array = 255.0 - img_array
                
                # Normalize and reshape
                img_input = img_array.astype('float32') / 255.0
                img_input = img_input.reshape(1, 28, 28, 1)

                # Predict
                model = st.session_state.model
                prediction = model.predict(img_input, verbose=0)
                predicted_digit = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100
                
                # Store results in session state to display in other column
                st.session_state.pred_digit = predicted_digit
                st.session_state.pred_conf = confidence
                st.session_state.probs = prediction[0]

    with col_res:
        st.markdown("**Result**")
        
        if 'pred_digit' in st.session_state:
            # Display big number
            st.markdown(f"""
            <div class="pred-box">
                <div style="font-size: 0.9rem; color: #b0b0d0; margin-bottom: 5px;">PREDICTED DIGIT</div>
                <div class="digit">{st.session_state.pred_digit}</div>
                <div style="font-size: 1.2rem; color: #a78bfa; margin-top: 5px;">
                    {st.session_state.pred_conf:.1f}% Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Plotly Chart
            probs = st.session_state.probs
            colors = ["#f093fb" if i == st.session_state.pred_digit else "#667eea" for i in range(10)]
            
            fig = go.Figure(data=[
                go.Bar(x=list(range(10)), y=probs, marker_color=colors, text=[f"{p:.2f}" for p in probs], textposition='outside')
            ])
            fig.update_layout(
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                font=dict(color="#ffffff"),
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, color="#b0b0d0"),
                yaxis=dict(showgrid=True, gridcolor="#2d2d4e", color="#b0b0d0", range=[0, 1]),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("""
            <div class="pred-box" style="opacity: 0.5;">
                <div style="font-size: 4rem;">?</div>
                <div style="color: #b0b0d0; margin-top: 10px;">Draw on the canvas and click Predict</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Placeholder when not trained
    st.markdown("<div class='card' style='text-align:center; opacity:0.5;'>", unsafe_allow_html=True)
    st.markdown("### 🎨 Drawing Canvas Locked")
    st.markdown("Complete **Step 1** to unlock this section.")
    st.markdown("</div>", unsafe_allow_html=True)
