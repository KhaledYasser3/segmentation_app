import streamlit as st
import os
import cv2
from inference import load_model, run_segmentation

st.set_page_config(page_title="Medical Segmentation", layout="centered")

st.title("Capsule Endoscopy Segmentation")
st.write("Upload an image and get the lesion mask.")

# ===== Load model once =====
@st.cache_resource
def load_seg_model():
    return load_model("best_model.pth")

model = load_seg_model()

# ===== Upload =====
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    os.makedirs("tmp", exist_ok=True)

    input_path = f"tmp/{uploaded_file.name}"
    output_path = f"tmp/mask_{uploaded_file.name}"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(input_path, caption="Original Image", use_column_width=True)

    if st.button("Run Segmentation"):
        with st.spinner("Running segmentation..."):
            run_segmentation(model, input_path, output_path)

        st.success("Done!")
        st.image(output_path, caption="Predicted Mask", use_column_width=True)
