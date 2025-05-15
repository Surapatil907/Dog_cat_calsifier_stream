import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

MODEL_PATH = 'dog_cat_model.h5'
MODEL_URL = 'https://drive.google.com/uc?id=1fjkr8DNhQJObvSunnXaiXcjex-Nwz1z_'

# Check if model exists locally, if not download
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model from Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success('Model downloaded successfully!')

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# App title
st.title("🐶😺 Dog vs Cat Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image of Dog or Cat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_container_width=True)

    # Preprocess
    img = img.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success(f"Prediction: Dog ({prediction[0][0]:.2f})")
    else:
        st.success(f"Prediction: Cat ({1 - prediction[0][0]:.2f})")
