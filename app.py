import streamlit as st
import cv2
import numpy as np
from predict import predict_sign

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered")

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image to detect and get a voice alert.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Sign"):
        sign_name = predict_sign(img)
        st.success(f"Detected Sign: **{sign_name}**")
