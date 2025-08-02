import streamlit as st
import cv2
import numpy as np
import joblib
import os
from datetime import datetime

# Load Model

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please make sure model.pkl exists.")
else:
    model = joblib.load(MODEL_PATH)

# Classes
CLASSES = ['cut', 'burn', 'abrasions', 'normal skin']

st.title("🩹 DermAI - Wound Detection")
st.write("Upload an image or capture from webcam to detect wounds.")


# Feature Extraction

def extract_features_from_array(img_array):
    """Extract features from image array for prediction."""
    img = cv2.resize(img_array, (64, 64))  # resize to match model input
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to HSV
    features = hsv.flatten()  # flatten to 1D vector
    return features

def predict_image_array(img_array):
    """Predict class from image array."""
    features = extract_features_from_array(img_array).reshape(1, -1)
    prediction = model.predict(features)[0]
    return CLASSES[prediction]


# File Upload Section

st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Failed to load image. Please check the file format.")
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
        prediction = predict_image_array(img)
        st.success(f"### 🧾 Prediction: **{prediction}**")

# Live Camera Capture

st.subheader("📷 Capture from Webcam")
capture_btn = st.button("Start Live Camera")

if capture_btn:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check camera access.")
    else:
        st.info("Press **Q** to quit the webcam window.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Show webcam feed
            cv2.imshow("Press Q to Quit", frame)

            # Prediction
            prediction = predict_image_array(frame)
            cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display prediction on live window
            cv2.imshow("Press Q to Quit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

