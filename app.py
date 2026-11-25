import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Lung X-Ray Classifier", page_icon="🫁")

CLASS_NAMES = ["Lung_Opacity", "Normal", "Viral Pneumonia"]
MODEL_PATH = Path(__file__).parent / "lung_model_v1.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("🫁 Lung Disease Classification from Chest X-Rays")
st.write("Upload a chest X-ray image and the model will predict one of three classes: "
         "Normal, Lung Opacity or Viral Pneumonia.")

with st.spinner("Loading model, please wait..."):
    model = load_model()

uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    st.subheader(f"Prediction: {CLASS_NAMES[predicted_index].replace('_', ' ')}")
    st.write(f"Confidence: {probabilities[predicted_index] * 100:.1f}%")

    st.write("Probability for each class:")
    for name, prob in zip(CLASS_NAMES, probabilities):
        st.write(f"{name.replace('_', ' ')}: {prob * 100:.1f}%")
        st.progress(float(prob))

st.caption("⚠️ This is a student project for learning purposes. "
           "It is not a medical diagnosis tool.")