import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="Flower Classification", layout="wide")

# Load Model
@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model('model/flower_rnn.keras')

model = load_flower_model()
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def preprocess_image(uploaded_file):
    """Preprocess uploaded image to match the 128x128 model input tensor."""
    img = Image.open(uploaded_file).convert('RGB')
    
    # Updated target size to 128x128
    img = img.resize((128, 128))
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor

# Main Interface
st.title("Flower Classification")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, use_container_width=True)
        
    input_data = preprocess_image(uploaded_file)
    predictions = model.predict(input_data, verbose=0)[0]
    
    top_class_idx = np.argmax(predictions)
    top_class = CLASS_NAMES[top_class_idx].capitalize()
    top_confidence = predictions[top_class_idx] * 100

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