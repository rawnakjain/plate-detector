import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="License Plate Detector", layout="centered")

st.title("🚗 License Plate Detection")
st.write("Upload an image with vehicles to detect license plate-like regions (using classical image processing, not ML).")

# ---- Upload Image ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # BGR
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

    # --- Processing pipeline ---
    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def detect_edges(image):
        edges = cv2.Canny(image, 100, 200)
        return edges

    def find_candidate_regions(edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.0 <= aspect_ratio <= 5.0 and w > 60 and h > 15:
                area = cv2.contourArea(cnt)
                if area > 1000:
                    candidates.append((x, y, w, h))
        return candidates

    def draw_bounding_boxes(image, regions):
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

    # --- Run pipeline ---
    preprocessed = preprocess_image(image)
    edges = detect_edges(preprocessed)
    regions = find_candidate_regions(edges)
    output_img = draw_bounding_boxes(image.copy(), regions)

    # Convert BGR to RGB for display
    st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(regions)-1} regions", use_container_width=True)

    # Optional: Download link
    _, buffer = cv2.imencode(".jpg", output_img)
    st.download_button("📥 Download Result", buffer.tobytes(), file_name="output.jpg", mime="image/jpeg")
