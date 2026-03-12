import streamlit as st
import streamlit_drawable_canvas as canvas
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import zipfile

# --- 1. Config ---
st.set_page_config(page_title="AI Digit Dashboard", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    h1, h2, h3, label { color: white !important; }
    div[data-testid="stMetricValue"] { font-size: 5rem; color: #ff4c4c; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #b8b8b8; font-size: 1.2rem; }
    .stButton>button { width: 100%; height: 3em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. Sidebar: Debugging & Settings ---
st.sidebar.header("⚙️ Settings")

# Invert Training Data Checkbox (Agar CSV mein Black digit hai, toh ye ON karein)
invert_training_data = st.sidebar.checkbox("Invert Training Data (Black Digit Fix)", value=False, help="Agar CSV mein images Black on White hain, toh ye tick karein.")

# --- 3. Model Logic ---
MODEL_PATH = 'mnist_debug_model.h5'
ZIP_FILE = 'train.csv.zip'

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH), None # Return None for sample image if loaded

    if not os.path.exists(ZIP_FILE):
        st.error(f"❌ '{ZIP_FILE}' nahi mili.")
        return None, None

    with st.spinner(f"⏳ Processing {ZIP_FILE} & Training..."):
        try:
            with zipfile.ZipFile(ZIP_FILE, 'r') as z:
                file_list = z.namelist()
                csv_file_name = next((f for f in file_list if f.endswith('.csv')), None)
                
                if not csv_file_name:
                    st.error("ZIP mein CSV nahi mili.")
                    return None, None

                with z.open(csv_file_name) as f:
                    # IMPORTANT: header=None maan rahe hain. Agar headings hain toh ye shayad galat lega,
                    # lekin hum preview se check karenge.
                    df = pd.read_csv(f, header=None) 

            # Data Separation
            # Assume: Col 0 = Label, Col 1... = Pixels
            y_data = df.iloc[:, 0].values
            x_data = df.iloc[:, 1:].values 

            # --- FIX 1: Invert Logic (If User Selected) ---
            # Agar CSV Black/White hai (Digit=0, BG=255), aur model White/Black chahiye, toh invert karein.
            if invert_training_data:
                x_data = 255 - x_data

            # Reshape
            x_data = x_data.reshape(-1, 28, 28, 1)
            
            # Save 1 Sample for Sidebar Preview
            sample_img = x_data[0].reshape(28, 28)

            # Normalize
            x_data = x_data / 255.0

            # Train/Test Split
            split = int(0.8 * len(x_data))
            x_train, x_test = x_data[:split], x_data[split:]
            y_train, y_test = y_data[:split], y_data[split:]

            # Model
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
            model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)
            
            model.save(MODEL_PATH)
            st.success("✅ Model Train & Save ho gaya!")
            
            return model, sample_img
            
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None

# Load Model
model, sample_img = get_model()

# --- SIDEBAR: PREVIEW (Important for Debugging) ---
st.sidebar.subheader("🔍 Training Data Check")
if model is not None:
    st.sidebar.write("Is image mein digit saaf dikh raha hai?")
    if sample_img is not None:
        # Display the first image from your CSV
        st.sidebar.image(sample_img, width=150, clamp=True)
    else:
        st.sidebar.caption("(Model purana load hua hai, preview nahi dikhega)")
    
    st.sidebar.caption("Agar yeh image 'kala' (black) hai aur background 'safed' (white), toh upar 'Invert Training Data' ko tick karein aur model delete karke wapas run karein.")

# --- 4. Main App ---
st.title("🖊️ Handwritten Digit Classifier")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✏️ Canvas")
    canvas_result = canvas.st_canvas(
        fill_color="#ffffff",
        stroke_width=12,
        stroke_color="#000000", # Default Black
        background_color="#ffffff",
        drawing_mode="freedraw",
        key="canvas_main"
    )

with col2:
    st.subheader("🔍 Result")
    metric_placeholder = st.empty()
    conf_placeholder = st.empty()

st.markdown("---")

if st.button("🚀 PREDICT", type="primary"):
    if model is None:
        st.error("Model load nahi hui.")
    elif canvas_result.image_data is not None:
        with st.spinner("Analyzing..."):
            # Image Processing
            img_data = canvas_result.image_data
            img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img)
            
            # Invert (Canvas Black/White -> Model White/Black)
            img_array = 255 - img_array 
            img_array = img_array / 255.0
            
            # Thresholding
            img_array = (img_array > 0.4).astype('float32')
            
            img_input = img_array.reshape(1, 28, 28, 1)
            
            prediction = model.predict(img_input)
            digit = np.argmax(prediction)
            conf = np.max(prediction)
            
            metric_placeholder.metric("Detected", int(digit))
            conf_placeholder.write(f"Confidence: {conf*100:.2f}%")

            df = pd.DataFrame({'Digit': range(10), 'Probability': (prediction[0] * 100).round(2)})
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index('Digit'))
    else:
        st.error("Draw something first!")

if st.button("🗑️ Clear"):
    st.rerun()
