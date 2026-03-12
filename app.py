import streamlit as st
import streamlit_drawable_canvas as canvas
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- 1. Config ---
st.set_page_config(
    page_title="AI Digit Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    h1, h2, h3, label { color: white !important; }
    div[data-testid="stMetricValue"] { font-size: 5rem; color: #ff4c4c; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #b8b8b8; font-size: 1.2rem; }
    .stButton>button { width: 100%; height: 3em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. Model Logic ---
MODEL_PATH = 'mnist_model.h5'

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    
    with st.spinner("⏳ Training Model... (First Run)"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), verbose=0)
        model.save(MODEL_PATH)
        st.success("✅ Model Saved!")
    return model

model = get_model()

# --- 3. Sidebar ---
st.sidebar.header("🛠️ Tools")
stroke_width = st.sidebar.slider("Width", 1, 30, 12)
stroke_color = st.sidebar.color_picker("Color", "#000000")
bg_color = st.sidebar.color_picker("Background", "#ffffff")

# --- 4. Main Layout ---
st.title("🖊️ Handwritten Digit Classifier")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✏️ Canvas")
    
    # FIXED CANVAS CODE
    canvas_result = canvas.st_canvas(
        fill_color="#ffffff",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        drawing_mode="freedraw", # Hardcoded for safety
        key="canvas_fixed"
    )

with col2:
    st.subheader("🔍 Result")
    metric_placeholder = st.empty()
    conf_placeholder = st.empty()

# --- 5. Predict ---
st.markdown("---")
st.subheader("📊 Prediction Table")

if st.button("🚀 PREDICT", type="primary"):
    if canvas_result.image_data is not None:
        with st.spinner("Analyzing..."):
            # Image Processing
            img_data = canvas_result.image_data
            img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img)
            
            # Invert & Normalize
            img_array = 255 - img_array
            img_array = img_array / 255.0
            img_input = img_array.reshape(1, 28, 28, 1)
            
            # Predict
            prediction = model.predict(img_input)
            digit = np.argmax(prediction)
            conf = np.max(prediction)
            
            # Update UI
            metric_placeholder.metric("Detected", int(digit))
            conf_placeholder.write(f"Confidence: {conf*100:.2f}%")
            
            # Table
            df = pd.DataFrame({
                'Digit': range(10),
                'Probability': (prediction[0] * 100).round(2)
            })
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index('Digit'))
    else:
        st.error("Draw something first!")

if st.button("🗑️ Clear"):
    st.rerun()
