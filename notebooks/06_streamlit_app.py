# 06_streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import pathlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Leaf Disease Detector", page_icon="🌱")


# ==============================
# Paths
# ==============================
MODEL_PATH = r"C:\plant_detection\models\leafnet_v1\best_model.h5"
CLASS_NAMES_PATH = r"C:\plant_detection\models\class_names.json"
IMG_SIZE = (224, 224)

# ==============================
# Load model + class names
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# Extract supported plant names
def extract_plant_name(label: str):
    # Split on ___ first, if not found, split on _
    if "___" in label:
        return label.split("___")[0]
    else:
        return label.split("_")[0]

supported_plants = sorted(set(extract_plant_name(c) for c in class_names))

# ==============================
# Helper functions
# ==============================
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    return arr

def predict_topk(pil_img: Image.Image, top_k=3):
    x = preprocess_image(pil_img)
    preds = model.predict(x, verbose=0)[0]
    topk_idx = np.argsort(preds)[-top_k:][::-1]
    return [(class_names[i], float(preds[i])) for i in topk_idx]

def parse_prediction(label: str):
    plant = extract_plant_name(label)
    if "healthy" in label.lower():
        return plant, "Yes", "None"
    else:
        disease = label.replace(plant, "").replace("___", "").replace("_", " ").strip()
        return plant, "No", disease

# ==============================
# Streamlit UI
# ==============================

st.title("🌱 Leaf Disease Detector")
st.write("Upload a leaf photo to detect plant diseases using a trained deep learning model.")

# Step 1: Plant name input
plant_input = st.text_input("Enter the plant name (e.g. Tomato, Potato, Pepper__bell):")

if plant_input:
    if plant_input not in supported_plants:
        st.error(
            f"❌ The model cannot predict for **{plant_input}**.\n\n"
            f"✅ Supported plants are: {', '.join(supported_plants)}"
        )
    else:
        # Step 2: Upload image
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Prediction
            top_preds = predict_topk(image, top_k=3)
            best_label, best_prob = top_preds[0]

            # Parse result
            plant, healthy, disease = parse_prediction(best_label)

            # Display result
            st.subheader("🔍 Prediction Result")
            st.write(f"**Plant name:** {plant}")
            st.write(f"**Healthy:** {healthy}")
            st.write(f"**Disease:** {disease}")
            st.write(f"**Confidence:** {best_prob * 100:.2f}%")

            # Bar chart for top-3 predictions
            st.subheader("📊 Top-3 Predictions")
            labels = [lbl for lbl, _ in top_preds]
            probs = [p * 100 for _, p in top_preds]

            fig, ax = plt.subplots()
            ax.barh(labels[::-1], probs[::-1], color="green")
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            for i, v in enumerate(probs[::-1]):
                ax.text(v + 1, i, f"{v:.2f}%", va="center")
            st.pyplot(fig)
