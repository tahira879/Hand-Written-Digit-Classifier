import streamlit as st
import streamlit_drawable_canvas as canvas
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ------------------------------------------------------------
# 1. Premium Dashboard Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Digit Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for Styling (Dark Theme)
st.markdown("""
<style>
    /* Background Colors */
    .stApp { background-color: #0e1117; color: white; }
    
    /* Text Colors */
    h1, h2, h3, label, .stMarkdown { color: white !important; }
    
    /* Metric Card Styling (Big Result) */
    div[data-testid="stMetricValue"] { 
        font-size: 6rem; 
        color: #ff4c4c; 
        font-weight: 800; 
        text-align: center;
    }
    div[data-testid="stMetricLabel"] { 
        color: #b8b8b8; 
        font-size: 1.2rem; 
        text-align: center;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 2. AI Model Logic (Load or Train)
# ------------------------------------------------------------
MODEL_PATH = 'mnist_digit_model.h5'

@st.cache_resource
def get_model():
    # Agar model save hai toh use karein
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    
    # Agar model nahi hai toh train karein (First Run)
    with st.spinner("⏳ First Run: Training Model... (Please wait 1-2 mins)"):
        # Data Load (MNIST)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Preprocessing
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., tf.newaxis] # (28, 28, 1)
        x_test = x_test[..., tf.newaxis]

        # CNN Model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax') # Output 0-9
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train (3 epochs for speed)
        model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), verbose=0)
        
        # Save Model
        model.save(MODEL_PATH)
        st.success("✅ Model Trained & Saved Successfully!")
    return model

model = get_model()

# ------------------------------------------------------------
# 3. Sidebar Controls (Tools)
# ------------------------------------------------------------
st.sidebar.header("🛠️ Drawing Tools")

# Pencil Width
stroke_width = st.sidebar.slider("Pencil Width", 1, 30, 12)

# Palette (Color Picker)
stroke_color = st.sidebar.color_picker("🎨 Pick Color", "#000000")
st.sidebar.caption("Tip: Eraser ke liye White color select karein.")

# Background Color
bg_color = st.sidebar.color_picker("Canvas Background", "#ffffff")

# Drawing Mode
drawing_mode = st.sidebar.selectbox("Mode", ("freedraw", "transform", "line", "rect", "circle"), index=0)

st.sidebar.markdown("---")
realtime_update = st.sidebar.checkbox("Update in Realtime", value=True)

# ------------------------------------------------------------
# 4. Main Layout (Canvas + Result)
# ------------------------------------------------------------
st.title("🖊️ Handwritten Digit Classifier")

# Create Columns (Left: Canvas, Right: Result)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✏️ Canvas")
    # Drawable Canvas Component
    canvas_result = canvas.st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=450,
        height=450,
        drawing_mode=drawing_mode,
        key="canvas_main",
        realtime_update=realtime_update
    )

with col2:
    st.subheader("🔍 Result")
    # Placeholders to show result later
    metric_placeholder = st.empty()
    conf_placeholder = st.empty()
    status_placeholder = st.empty()

# ------------------------------------------------------------
# 5. Prediction Logic & Bottom Table
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Prediction Table")

if st.button("🚀 PREDICT", type="primary", use_container_width=True):
    if canvas_result.image_data is not None:
        with st.spinner("Analyzing..."):
            # A. Image Preprocessing
            img_data = canvas_result.image_data
            img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            
            # Convert to Grayscale
            img = img.convert('L')
            
            # Resize to 28x28 (MNIST size)
            img = img.resize((28, 28))
            
            # Convert to Array
            img_array = np.array(img)
            
            # Invert Colors (Canvas: White BG, Model: Black BG)
            img_array = 255 - img_array
            
            # Normalize (0 to 1)
            img_array = img_array / 255.0
            
            # Reshape for Model (1, 28, 28, 1)
            img_input = img_array.reshape(1, 28, 28, 1)
            
            # B. Prediction
            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # C. Update Result Column (Col 2)
            metric_placeholder.metric(label="Detected Digit", value=int(predicted_digit))
            conf_placeholder.write(f"**Confidence:** {confidence*100:.2f}%")
            
            if confidence > 0.8:
                status_placeholder.success("High Confidence ✅")
            elif confidence > 0.5:
                status_placeholder.warning("Medium Confidence ⚠️")
            else:
                status_placeholder.error("Low Confidence ❌")

            # D. Update Bottom Table
            # Create DataFrame for all 10 digits
            probs = prediction[0]
            df = pd.DataFrame({
                'Digit': [i for i in range(10)],
                'Probability (%)': [p * 100 for p in probs]
            })
            
            # Show Table
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Show Bar Chart
            st.bar_chart(df.set_index('Digit'))

    else:
        st.error("⚠️ Canvas par kuch draw karein pehle!")

# Clear Button
if st.button("🗑️ Clear Canvas", use_container_width=True):
    st.rerun()
