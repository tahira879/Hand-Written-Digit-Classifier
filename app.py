import streamlit as st
import streamlit_drawable_canvas as canvas
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import zipfile
import matplotlib.pyplot as plt

# --- 1. Config ---
st.set_page_config(page_title="AI Digit Dashboard", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    h1, h2, h3, label { color: white !important; }
    div[data-testid="stMetricValue"] { font-size: 4rem; color: #ff4c4c; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #b8b8b8; font-size: 1rem; }
    .stButton>button { width: 100%; height: 3em; font-weight: bold; }
    /* Image containers styling */
    .img-container { text-align: center; border: 1px solid #444; padding: 10px; border-radius: 10px; background: #000; }
</style>
""", unsafe_allow_html=True)

# --- 2. Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
st.sidebar.warning("Agar prediction galat hai, toh 'Invert' ko change karke model delete karein aur wapas run karein.")

invert_training_data = st.sidebar.checkbox("Invert Training Data (Fix Black Digits)", value=False)

# --- 3. Model Logic ---
MODEL_PATH = 'mnist_debug_v2.h5'
ZIP_FILE = 'train.csv.zip'

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH), None

    if not os.path.exists(ZIP_FILE):
        st.error(f"❌ '{ZIP_FILE}' nahi mili!")
        return None, None

    with st.spinner(f"⏳ Training from {ZIP_FILE}..."):
        try:
            with zipfile.ZipFile(ZIP_FILE, 'r') as z:
                file_list = z.namelist()
                csv_file_name = next((f for f in file_list if f.endswith('.csv')), None)
                
                if not csv_file_name:
                    st.error("ZIP mein CSV nahi mili.")
                    return None, None

                with z.open(csv_file_name) as f:
                    df = pd.read_csv(f, header=0) 

            df = df.astype('float32')
            y_data = df.iloc[:, 0].values
            x_data = df.iloc[:, 1:].values 

            # Apply Invert if needed
            if invert_training_data:
                x_data = 255 - x_data
                st.info("Inverted Training Data applied.")

            x_data = x_data.reshape(-1, 28, 28, 1)
            sample_img = x_data[0].reshape(28, 28)
            
            x_data = x_data / 255.0

            split = int(0.8 * len(x_data))
            x_train, x_test = x_data[:split], x_data[split:]
            y_train, y_test = y_data[:split], y_data[split:]

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
            model.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test), verbose=0)
            
            model.save(MODEL_PATH)
            st.success("✅ Model Trained!")
            return model, sample_img
            
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None

model, sample_img = get_model()

# --- SIDEBAR PREVIEW ---
st.sidebar.subheader("📊 Training Data Preview")
if sample_img is not None:
    st.sidebar.image(sample_img, width=120, caption="CSV ka 1st Image (Centered)")
    st.sidebar.caption("Yeh White digit hona chahiye.")

# --- 4. Main UI ---
st.title("🔍 Visual Debug Dashboard")

col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    st.subheader("1. Draw Here")
    canvas_result = canvas.st_canvas(
        fill_color="#ffffff",
        stroke_width=20, # Moti pencil
        stroke_color="#000000",
        background_color="#ffffff",
        drawing_mode="freedraw",
        height=300,
        width=300,
        key="canvas_main"
    )

with col2:
    st.subheader("2. Model Input (28x28)")
    st.caption("Yeh wo image hai jo model ko mil rahi hai.")
    model_view_placeholder = st.empty()

with col3:
    st.subheader("3. Result")
    metric_placeholder = st.empty()
    conf_placeholder = st.empty()
    status_placeholder = st.empty()

st.markdown("---")

if st.button("🚀 PREDICT & DEBUG", type="primary"):
    if model is None:
        st.error("Model load nahi hui.")
    elif canvas_result.image_data is not None:
        with st.spinner("Analyzing..."):
            # --- PROCESS DRAWING ---
            img_data = canvas_result.image_data
            img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img)
            
            # Invert for Model (Canvas Black/White -> Model White/Black)
            img_array = 255 - img_array 
            img_array = img_array / 255.0
            
            # Thresholding
            img_array = (img_array > 0.3).astype('float32')
            
            # Prepare Input
            img_input = img_array.reshape(1, 28, 28, 1)
            
            # --- SHOW MODEL VIEW (Debugging) ---
            # 28x28 image ko wapas display ke liye prepare karein
            display_img = img_array.reshape(28, 28)
            # Colormap use karein taaki clearly dikhe
            model_view_placeholder.image(display_img, width=150, clamp=True, caption="What Model Sees")

            # --- PREDICT ---
            prediction = model.predict(img_input)
            digit = np.argmax(prediction)
            conf = np.max(prediction)
            
            metric_placeholder.metric("Detected", int(digit))
            conf_placeholder.write(f"Conf: {conf*100:.1f}%")
            
            # Diagnostic Message
            if conf > 0.8:
                status_placeholder.success("Model sure hai.")
            else:
                status_placeholder.warning("Model confuse hai. Check 'Model View'.")

            # --- COMPARISON CHECK ---
            st.subheader("🔎 Comparison Check")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**What You Drew (Original)**")
                st.image(img_data, width=150)
            with c2:
                st.write("**What Model Sees (Processed)**")
                st.image(display_img, width=150, clamp=True)

            st.info("💡 **Tip:** Agar 'What Model Sees' mein **Black** digit hai, toh Sidebar mein **'Invert Training Data'** ko tick karein, model delete karein aur restart karein.")

    else:
        st.error("Pehle draw karein!")

if st.button("🗑️ Clear"):
    st.rerun()
