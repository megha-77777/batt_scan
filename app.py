import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Battery CT Scanner", layout="centered")

st.title("🔋 Lithium Battery CT Scanner (IR Vision + AI)")
st.markdown("Upload an IR image of the battery to detect cracks, bulging, and revival potential.")

model = load_model("battery_model.h5")

uploaded_file = st.file_uploader("📸 Upload IR Battery Image", type=["jpg", "png"])

if uploaded_file:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded IR Image", use_column_width=True)

    resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).reshape(1, 64, 64, 1) / 255.0
    pred = model.predict(gray)
    label = np.argmax(pred)

    labels = ["Healthy ✅", "Bulging ⚠️", "Cracked ❌"]
    st.subheader(f"🧠 AI Diagnosis: {labels[label]}")
