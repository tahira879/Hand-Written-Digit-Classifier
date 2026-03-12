import streamlit as st
import streamlit_drawable_canvas as canvas
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- 1. Premium Dashboard Config ---
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

# --- 2. AI Model Logic (Load CSV & Train) ---
MODEL_PATH = 'mnist_csv_model.h5'
CSV_FILE = 'mnist_dataset (1).csv' # Aapki file ka naam

@st.cache_resource
def get_model():
    # 1. Check agar model pehle se save hai
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Agar model nahi hai, toh CSV se train karein
    if not os.path.exists(CSV_FILE):
        st.error(f"❌ Error: '{CSV_FILE}' file nahi mili! Isse app.py ke saath rakhna.")
        return None

    with st.spinner(f"⏳ Training Model from {CSV_FILE}... (Thaba time lega)"):
        try:
            # A. CSV Load karein
            df = pd.read_csv(CSV_FILE)
            
            # B. Data Preprocessing
            # Maan rahe hain ki 0th column 'Label' hai aur baaki 'Pixels' hain
            # Agar aapki CSV mein Label last mein hai, toh .iloc[:, 0] ko .iloc[:, -1] se replace karein
            y_data = df.iloc[:, 0].values
            x_data = df.iloc[:, 1:].values 

            # Reshape: Flat pixels (784) ko Image (28, 28, 1) mein convert karein
            x_data = x_data.reshape(-1, 28, 28, 1)
            
            # Normalize (0-1)
            x_data = x_data / 255.0

            # C. Train/Test Split (Simple 80-20 split without sklearn)
            split_index = int(0.8 * len(x_data))
            x_train, x_test = x_data[:split_index], x_data[split_index:]
            y_train, y_test = y_data[:split_index], y_data[split_index:]

            # D. CNN Model Build
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10, activation='softmax') # 0-9 Output
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # E. Model Train (3-5 epochs)
            model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)
            
            # F. Save Model
            model.save(MODEL_PATH)
            st.success(f"✅ Model {CSV_FILE} se train ho gaya aur save ho gaya!")
            
        except Exception as e:
            st.error(f"CSV Load ya Train karne mein error aaya: {e}")
            return None
            
    return model

model = get_model()

# --- 3. Sidebar Controls ---
st.sidebar.header("🛠️ Tools")
stroke_width = st.sidebar.slider("Width", 1, 30, 12)
stroke_color = st.sidebar.color_picker("Color", "#000000")
bg_color = st.sidebar.color_picker("Background", "#ffffff")

# --- 4. Main Layout ---
st.title("🖊️ Handwritten Digit Classifier")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✏️ Canvas")
    canvas_result = canvas.st_canvas(
        fill_color="#ffffff",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        drawing_mode="freedraw",
        key="canvas_main"
    )

with col2:
    st.subheader("🔍 Result")
    metric_placeholder = st.empty()
    conf_placeholder = st.empty()
    status_placeholder = st.empty()

# --- 5. Prediction Logic ---
st.markdown("---")
st.subheader("📊 Prediction Table")

if st.button("🚀 PREDICT", type="primary"):
    if model is None:
        st.error("Model load nahi hui. CSV file check karein.")
    elif canvas_result.image_data is not None:
        with st.spinner("Analyzing..."):
            # Image Processing
            img_data = canvas_result.image_data
            img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img)
            
            # Invert Colors
            img_array = 255 - img_array 
            img_array = img_array / 255.0
            
            # Thresholding (Fix for multi-color predictions)
            img_array = (img_array > 0.4).astype('float32')
            
            # Reshape
            img_input = img_array.reshape(1, 28, 28, 1)
            
            # Predict
            prediction = model.predict(img_input)
            digit = np.argmax(prediction)
            conf = np.max(prediction)
            
            # Update UI
            metric_placeholder.metric("Detected", int(digit))
            conf_placeholder.write(f"Confidence: {conf*100:.2f}%")
            
            if conf > 0.8:
                status_placeholder.success("High Confidence ✅")
            elif conf > 0.5:
                status_placeholder.warning("Medium Confidence ⚠️")
            else:
                status_placeholder.error("Low Confidence ❌")

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
