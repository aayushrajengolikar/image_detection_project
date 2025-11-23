# app_detection.py
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO Model
# -----------------------------
st.title("Object Detection App")
yolo_model = YOLO("best.pt")  # updated path

# -----------------------------
# YOLO Prediction
# -----------------------------
def predict_objects_yolo(image: Image.Image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = yolo_model(img)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_model.names[int(cls)]} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# -----------------------------
# Streamlit UI
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    st.write("Processing object detection...")
    detected_image = predict_objects_yolo(image)
    st.image(detected_image, caption="Detected Objects", width=500)
