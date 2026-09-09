import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "flower_rnn.keras"
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Page Config
st.set_page_config(page_title="Flower Classification", layout="wide")

# Load Model
@st.cache_resource
def load_flower_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train.py first.")
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_flower_model()
except Exception as exc:
    model = None
    st.warning(f"Model could not be loaded: {exc}")


def preprocess_image(uploaded_file):
    """Preprocess uploaded image to match the model input tensor."""
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# Main Interface
st.title("Flower Classification")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("The model is not available. Please retrain the model before predicting.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    input_data = preprocess_image(uploaded_file)
    predictions = model.predict(input_data, verbose=0)[0]

    top_class_idx = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_class_idx].capitalize()
    top_confidence = float(predictions[top_class_idx]) * 100

    with col2:
        st.subheader("Prediction Results")
        st.success(f"**Predicted: {top_class}**")
        st.write("### Confidence")
        st.title(f"{top_confidence:.2f}%")

    st.markdown("---")

    st.subheader("Confidence Scores Across All Classes")
    for i, class_name in enumerate(CLASS_NAMES):
        score = float(predictions[i])
        st.write(f"**{class_name.capitalize()}** — `{score * 100:.2f}%`")
        st.progress(score)