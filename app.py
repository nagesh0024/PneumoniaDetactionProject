import numpy as np
import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model

#load the trained model
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# Load the trained model
model_03 = load_model("pneumonia_vgg19_model.h5")

# Image preprocessing
def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (224, 224))  # Adjust size as per training
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def get_prediction(image):
    result = model_03.predict(image)[0]
    class_index = np.argmax(result)
    confidence = float(result[class_index]) * 100
    label = "Normal" if class_index == 0 else "Pneumonia"
    return label, confidence

# App title
st.markdown("<h1 style='text-align: center;'>🩺 Pneumonia Detection from Chest X-Ray</h1>", unsafe_allow_html=True)
st.write("Upload a chest X-ray image and the model will detect if it's **Normal** or shows signs of **Pneumonia**.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Chest X-Ray (Max 10MB)", type=["jpg", "png", "jpeg"])

# Max file size check
MAX_FILE_SIZE_MB = 10
if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error("🚫 File size exceeds 10MB. Please upload a smaller image.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

        with st.spinner("🔍 Analyzing X-ray..."):
            processed_img = preprocess_image(image)
            prediction, confidence = get_prediction(processed_img)

        st.success(f"🧠 **Prediction:** `{prediction}`")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")

        if prediction == "Pneumonia":
            st.warning("⚠️ Please consult a medical professional for further diagnosis.")
        else:
            st.balloons()
            st.write("✅ Lungs appear clear. Keep taking care of your health!")

# Footer
st.markdown("<hr><center>Made with ❤️ using Streamlit & TensorFlow</center>", unsafe_allow_html=True)
